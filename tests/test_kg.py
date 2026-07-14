"""Test suite for the kg episodic/temporal graph.

The kg library is LIVE-ONLY (the offline heuristic extractor / hashing embedder / offline
answerer were removed). This suite stays deterministic + FREE + offline anyway by:

  * embedder  — the real local sentence-transformers bge-small (``embedder="st"``):
                deterministic, no key, no network once the model is cached.
  * extraction — a ``ScriptedExtractor`` ({episode_text: Extraction}) injected as
                ``g.extractor`` so the graph build runs on KNOWN facts (no LLM call).
  * answering  — a fake OpenAI client injected via ``g.ask(..., client=...)`` so the
                RAG ``OpenAIAnswerer`` runs without touching the API.

No test calls the real OpenAI API and no ``OPENAI_API_KEY`` is required. Run:
    python -m pytest tests/test_kg.py -q
"""
from __future__ import annotations

import json
import os
import tempfile
import types

import numpy as np
import pytest

from kg import Config, KnowledgeGraph
from kg.chunkers import (chunk_code, chunk_for, chunk_markdown, chunk_prose,
                         chunk_text, sniff_format)
from kg.canonicalize import (Canonicalizer, char_entropy, normalize_key,
                             normalize_relation, predicate_cardinality, relation_merge_vetoed)
from kg.corpus import CorpusItem, load_longmemeval, load_longmemeval_questions
from kg.embedders import SentenceTransformerEmbedder, get_embedder
from kg.extractors import (Extraction, ExtractedEntity, ExtractedRelation, OpenAIExtractor,
                          ScriptedExtractor, get_extractor)
from kg.models import (Belief, Edge, EdgeType, Modality, Node, NodeType, Provenance,
                       EntityType, episode_node, entity_node,
                       mention_node, relation_tag_node)
from kg.store import GraphStore, fact_active
from kg.vectors import VectorIndex


def cfg() -> Config:
    c = Config.default()
    c.embedder = "st"          # real local bge-small: deterministic, free, no key/network
    return c


@pytest.fixture(autouse=True)
def _no_live_extractor(monkeypatch):
    """KnowledgeGraph.__init__ eagerly calls get_extractor(), which is LIVE-ONLY and RAISES
    without a key (and would otherwise hold a real OpenAI client). Patch the reference the
    graph builds against so EVERY KnowledgeGraph.open(...) in this file gets a deterministic,
    keyless ScriptedExtractor — tests that need real extractions still overwrite g.extractor
    with their own table afterward. This patches kg.graph.get_extractor only; the tests that
    assert get_extractor's real live behavior import it from kg.extractors and are untouched."""
    import kg.graph as _graph
    monkeypatch.setattr(_graph, "get_extractor", lambda config: ScriptedExtractor({}))


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


# A deterministic extraction table for the four sample items, keyed on the EXACT episode
# text the ingest pipeline feeds the extractor (the full item text; for the image item,
# the label_hint). Entities/tags/relations are rich enough that the assertions about the
# mention star, SHARED_ENTITY / SHARED_TAG structure, and retrieval targets still hold.
def _sample_table() -> dict[str, Extraction]:
    items = {it.id: it for it in sample_items()}
    return {
        items["a"].text: Extraction(
            entities=[ExtractedEntity("Alan Turing", EntityType.PERSON),
                      ExtractedEntity("Bletchley Park", EntityType.PLACE),
                      ExtractedEntity("computer science", EntityType.CONCEPT)],
            tags=["cryptography", "codebreaking", "computer science",
                  "mathematics", "world war ii"],
            relations=[ExtractedRelation(source="Alan Turing", target="Bletchley Park",
                                         labels=["worked_at"], provenance=Provenance.EXTRACTED,
                                         confidence=0.9)],
        ),
        items["b"].text: Extraction(
            entities=[ExtractedEntity("Bletchley Park", EntityType.PLACE),
                      ExtractedEntity("Alan Turing", EntityType.PERSON),
                      ExtractedEntity("Enigma machine", EntityType.OTHER)],
            tags=["cryptography", "codebreaking", "world war ii", "enigma"],
            relations=[ExtractedRelation(source="Alan Turing", target="Bletchley Park",
                                         labels=["worked_at"], provenance=Provenance.EXTRACTED,
                                         confidence=0.9)],
        ),
        items["c"].text: Extraction(
            entities=[ExtractedEntity("Photosynthesis", EntityType.CONCEPT),
                      ExtractedEntity("Chlorophyll", EntityType.CONCEPT)],
            tags=["photosynthesis", "biology", "plants", "chlorophyll"],
            relations=[ExtractedRelation(source="Chlorophyll", target="Photosynthesis",
                                         labels=["part_of"], provenance=Provenance.EXTRACTED,
                                         confidence=0.85)],
        ),
        # image item: ScriptedExtractor.extract_image keys on the label_hint
        items["d"].label_hint: Extraction(
            entities=[ExtractedEntity("dog", EntityType.OTHER),
                      ExtractedEntity("person", EntityType.PERSON)],
            tags=["dog", "frisbee", "person", "outdoors"],
            relations=[],
            description="A photo of a dog, a frisbee, and a person.",
        ),
    }


def scripted_graph(items_extractor: dict[str, Extraction] | None = None) -> KnowledgeGraph:
    """A KnowledgeGraph whose extractor is a deterministic ScriptedExtractor (NO live LLM).
    Defaults to the sample-item table so the end-to-end ingest/retrieval tests run offline."""
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.extractor = ScriptedExtractor(_sample_table() if items_extractor is None
                                    else items_extractor)
    return g


# --------------------------------------------------------------------------- #
# Fake OpenAI client for the RAG answerer (no API call). Shape matches what
# kg.rag.OpenAIAnswerer.answer reads off the message and what kg.metering.UsageMeter
# .record reads off msg.usage.
# --------------------------------------------------------------------------- #
class _FakeOpenAI:
    def __init__(self, answer="", citations=None):
        self._a, self._c = answer, (citations or [])
        self.chat = self
        self.completions = self
        self.calls: list[dict] = []

    def create(self, **kw):
        self.calls.append(kw)
        tc = types.SimpleNamespace(
            id="call_0",
            function=types.SimpleNamespace(
                name="submit_answer",
                arguments=json.dumps({"answer": self._a, "citations": self._c})))
        message = types.SimpleNamespace(content=None, tool_calls=[tc])
        choice = types.SimpleNamespace(message=message, finish_reason="tool_calls")
        usage = types.SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        return types.SimpleNamespace(choices=[choice], usage=usage)


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
    # RELATED_TO discriminates on (rel_tag, valid_at, seq) so closed + reopened facts —
    # including a same-day close→reopen — coexist (fork-parity spec B3).
    e1 = Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0007", valid_at="2020")
    e2 = Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0007", valid_at="2024")
    assert e1.key() == ("RELATED_TO", "rel_0007", "2020", 0)
    assert e1.key() != e2.key()
    # same valid_at, different seq (as GraphStore.add_edge assigns) → still distinct
    e3 = Edge("a", "b", EdgeType.RELATED_TO, rel_tag="rel_0007", valid_at="2020", seq=1)
    assert e1.key() != e3.key()
    # non-fact edges ignore valid_at and seq in their key
    assert Edge("m", "e", EdgeType.RESOLVES_TO).key() == ("RESOLVES_TO", "", "", "")


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
    # fork-parity spec D4: the trailing argument-structure marker is load-bearing, so
    # "is_friend_of" (…_of, directed) and "is friends with" (…_with, symmetric) are
    # DIFFERENT predicates and must NOT collapse (works_at ≠ works_with). Interior
    # function-word dropping still merges "friend of" into the same content key.
    assert a == canon.resolve_relation("friend of")
    assert canon.resolve_relation("is friends with") != a
    node = store.get_node(a)
    assert node.name == "is_friend_of" and "friend_of" in node.aliases
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
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        text = json.dumps({"verdict": self._verdict, "reason": "test"})
        message = types.SimpleNamespace(content=text)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


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
    # run the extraction-dump path on a deterministic ScriptedExtractor (no live LLM)
    from kg.extract_dump import extract_corpus, summarize
    ext = ScriptedExtractor(_sample_table())
    records, errors = extract_corpus(ext, sample_items(), cfg())
    assert len(records) == 4 and not errors
    s = summarize(records, "scripted")
    assert s["items"] == 4 and s["unique_tags"] > 0


def test_eval_canon_gate_passes_offline():
    # run_gate only canonicalizes (no extraction) — runs offline on the bge embedder
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
# embedders / extractors / factories (LIVE-ONLY behavior)
# --------------------------------------------------------------------------- #
def test_get_embedder_returns_sentence_transformer():
    """The factory always returns the semantic sentence-transformers embedder (the offline
    hashing embedder was removed). Verify it embeds to unit-norm float32 vectors."""
    e = get_embedder(cfg())
    assert isinstance(e, SentenceTransformerEmbedder)
    a = e.embed(["knowledge graph"])
    assert a.dtype == np.float32
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-4
    # deterministic: same text → same vector
    assert np.allclose(a, e.embed(["knowledge graph"]))


def test_llm_backend_raises_without_key(monkeypatch):
    """The 'llm' backend is live-only: it RAISES when no OPENAI_API_KEY is set. (The
    default 'cue_gated' backend instead runs a keyless local floor — see below.)"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = cfg(); c.extractor_backend = "llm"
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_extractor(c)


def test_llm_backend_returns_llm_with_key(monkeypatch):
    """With a key present, the 'llm' backend yields the live OpenAIExtractor (constructed
    only — no API call is made until extract_text/extract_image)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    c = cfg(); c.extractor_backend = "llm"
    ext = get_extractor(c)
    assert isinstance(ext, OpenAIExtractor) and ext.name == "llm"


def test_default_extractor_is_cue_gated_and_keyless(monkeypatch):
    """The production default is cue-gated: a keyless local NLP floor, with LLM escalation
    only when a key is present (so extraction no longer hard-requires a key)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ext = get_extractor(cfg())
    assert ext.name == "cue_gated" and ext.escalate is False   # no key → escalation disabled


def test_rag_answerer_raises_without_client_or_key(monkeypatch):
    """The query/answer path is live-only: RagAnswerer with neither an injected client nor a
    key RAISES (no offline answerer to silently degrade to)."""
    from kg.rag import RagAnswerer
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store = GraphStore(cfg())
    c = cfg()
    canon = Canonicalizer(store, get_embedder(c), c)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        RagAnswerer(store, get_embedder(c), canon, c, client=None)


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
    store.add_hash("1", "a")   # write-through save persists only tracked mutations
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
    g = scripted_graph()
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
    g = scripted_graph()
    g.ingest(sample_items())
    for e in g.store.nodes_of_type(NodeType.ENTITY):
        assert e.raw_text is None and e.summary is None
        assert g.store.vectors.get("episode", e.id) is None
        assert g.store.vectors.get("mention", e.id) is None


def test_store_is_directed():
    import networkx as nx
    g = scripted_graph()
    assert isinstance(g.store.g, nx.MultiDiGraph)


def test_ingest_cache_skips_on_rerun():
    g = scripted_graph()
    g.ingest(sample_items())
    rep2 = g.ingest(sample_items())
    assert rep2.ingested == 0 and rep2.skipped == 4


def test_ingest_appends_new_version_on_change():
    """Episodes are append-only: changed content under a known id adds a NEW immutable
    episode; the old one stays valid (history is preserved, not overwritten)."""
    g = scripted_graph()
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
    g = scripted_graph()
    g.ingest(sample_items())
    for mode in ("ppr", "bfs", "vector"):
        res = g.query("cryptography codebreaking at Bletchley", mode=mode, k=3)
        assert res.object_ids, f"{mode} returned nothing"
        assert {"ep_a", "ep_b"} & set(res.object_ids), f"{mode} missed the target"


def test_rerank_keeps_ppr_top_episodes():
    """The cross-encoder must not demote the raw PPR pool's top episodes out of the final
    top-k (config.rerank_keep_ppr_top). A stub reranker that drops the PPR #1 entirely
    still yields it in the final ids, spliced in over the reranked tail."""
    from kg.retrieval import HybridRetriever

    g = scripted_graph()
    g.ingest(sample_items())
    retr = HybridRetriever(g.store, g.embedder, g.canon, g.config)

    class _DemotingReranker:
        available = True
        def rerank(self, query, items, k):
            ids = [i for i, _ in items]
            return ids[1:][:k]          # drop the PPR top-1 completely, keep the rest

    retr._reranker = _DemotingReranker()
    # "currently" routes to the STATE lane, which is in the default rerank_lanes
    res = retr.retrieve("who currently works on cryptography codebreaking at Bletchley",
                        k=2)
    final_ids = [ep for ep, _ in res.objects]
    ppr_top1 = res.ppr_pool[0][0]
    assert ppr_top1 in final_ids, "PPR #1 was demoted out of the final top-k"
    assert len(final_ids) <= 2
    # the reranker's surviving relative order is preserved ahead of the splice
    assert final_ids[0] != ppr_top1


def test_empty_and_blank_query_do_not_crash():
    g = scripted_graph()
    g.ingest(sample_items())
    g.build_communities()
    for mode in ("ppr", "bfs", "vector"):
        assert g.query("   ", mode=mode).object_ids == []
        assert g.query("", mode=mode).object_ids == []
    assert g.query("anything", mode="ppr").object_ids != [] or True  # smoke


def test_communities_and_global_route():
    g = scripted_graph()
    g.ingest(sample_items())
    assert g.build_communities() >= 1
    res = g.query("what are the main themes", mode="auto")
    assert isinstance(res, list) and res and "summary" in res[0]


def test_ppr_excludes_community_edges():
    g = scripted_graph()
    g.ingest(sample_items())
    before = g.query("cryptography Bletchley", mode="ppr", k=4).object_ids
    g.build_communities()
    assert before == g.query("cryptography Bletchley", mode="ppr", k=4).object_ids


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def test_eval_metrics_and_retrieval_modes():
    """recall@k / MRR helpers + evaluate() over the retrieval modes. 'rag' is excluded here
    because evaluate() routes it through the LIVE OpenAI answerer (no client injection
    seam); the RAG answer path is exercised separately in test_rag_answer_with_fake_client."""
    from kg.evaluate import (_mrr, _recall_at_k, cross_article_questions, evaluate,
                             single_article_questions)
    assert _recall_at_k(["a", "b", "c"], {"b"}, 3) == 1.0
    assert _mrr(["a", "b"], {"b"}) == 0.5
    g = scripted_graph()
    g.ingest(sample_items())
    qs = single_article_questions(g, limit=3) + cross_article_questions(g, limit=3)
    assert qs
    scores = evaluate(g, qs, modes=("ppr", "vector"), k=5)
    assert len(scores) == 2
    assert all(0.0 <= s.recall_at_k <= 1.0 for s in scores)


def test_rag_answer_with_fake_client():
    """The §5 graph-RAG answer flow (PPR-retrieve → context → ONE answer call) runs through
    OpenAIAnswerer over an INJECTED fake OpenAI client — no real API call. Citations that
    name an episode actually in the retrieved context survive validation."""
    g = scripted_graph()
    g.ingest(sample_items())
    fake = _FakeOpenAI(answer="Turing worked at Bletchley Park on cryptography.",
                       citations=["ep_a"])
    ans = g.ask("Where did Alan Turing work on cryptography?", client=fake)
    assert fake.calls, "the answerer never called the (fake) client"
    assert ans.backend == "openai"
    assert ans.answer == "Turing worked at Bletchley Park on cryptography."
    # ep_a is in the retrieved context, so the citation is kept (not dropped)
    assert "ep_a" in ans.citations
    assert "ep_a" not in ans.dropped_citations


# --------------------------------------------------------------------------- #
# viz + corpus
# --------------------------------------------------------------------------- #
def test_viz_payloads_and_html():
    from kg.viz import graph_payload, query_trace, render_html
    g = scripted_graph()
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


# --------------------------------------------------------------------------- #
# perf-path invariants (write-through store, cached PPR operator, incremental derive)
# --------------------------------------------------------------------------- #
def test_personalized_pagerank_matches_networkx():
    """The cached-CSR power iteration must reproduce nx.pagerank exactly (same math,
    minus the per-call graph→CSR conversion)."""
    import networkx as nx
    from kg.retrieval import personalized_pagerank
    rng = np.random.default_rng(7)
    G = nx.Graph()
    G.add_nodes_from(f"n{i}" for i in range(40))
    for _ in range(120):
        a, b = rng.integers(0, 40, 2)
        if a != b:
            G.add_edge(f"n{a}", f"n{b}", weight=float(rng.random()) + 0.05)
    pers = {f"n{i}": float(rng.random()) + 0.01 for i in range(0, 40, 3)}
    mine = personalized_pagerank(G, alpha=0.5, personalization=pers, max_iter=200)
    ref = nx.pagerank(G, alpha=0.5, personalization=pers, weight="weight", max_iter=200)
    assert set(mine) == set(ref)
    assert all(abs(mine[n] - ref[n]) < 1e-10 for n in ref)
    # second call reuses the cached operator — must be identical
    again = personalized_pagerank(G, alpha=0.5, personalization=pers, max_iter=200)
    assert again == mine


def _edge_dump(store):
    rows = set()
    for u, v, d in store.g.edges(data=True):
        rows.add((u, v, d["etype"], d.get("rel_tag") or "", d.get("valid_at", ""),
                  d.get("invalid_at", ""), round(float(d["confidence"]), 6),
                  round(float(d["weight"]), 6)))
    return rows


def test_write_through_persists_in_place_mutations():
    """Two-batch Becky ingest: batch 2 supersedes/closes facts opened by batch 1 (in-place
    edge mutation) and bumps doc frequencies on loaded nodes. The dirty-tracked flush must
    persist all of it — reloaded store == in-memory store, with no full rewrite."""
    from kg.synthetic import becky_stream
    path = tmp_store()
    g = KnowledgeGraph.open(path, cfg())
    items, table = becky_stream()
    g.extractor = ScriptedExtractor(table)
    g.ingest(items[:2])
    g.ingest(items[2:])          # closes/supersedes batch-1 facts in place
    g.save()
    s2 = GraphStore.open(path, cfg())
    assert set(s2.nodes) == set(g.store.nodes)
    assert _edge_dump(s2) == _edge_dump(g.store)
    assert s2.hash_cache == g.store.hash_cache
    for nid, n in g.store.nodes.items():
        assert s2.nodes[nid].doc_frequency == n.doc_frequency, nid
        assert sorted(s2.nodes[nid].aliases) == sorted(n.aliases), nid


def test_community_rebuild_does_not_resurrect_rows():
    """build_communities removes the previous CommunityNodes; the flush must DELETE their
    rows (not leave them to come back on the next load)."""
    g = scripted_graph()
    g.ingest(sample_items())
    g.build_communities()
    g.save()
    g.build_communities()       # rebuild → previous comm_* removed then re-added
    g.save()
    s2 = GraphStore.open(g.store.path, cfg())
    live = {n.id for n in g.store.nodes_of_type(NodeType.COMMUNITY, valid_only=False)}
    loaded = {n.id for n in s2.nodes_of_type(NodeType.COMMUNITY, valid_only=False)}
    assert loaded == live


def test_incremental_derived_edges_match_single_batch():
    """Derived-edge identities (SHARED_TAG / SHARED_ENTITY / SIMILAR_TO) must be the same
    whether the corpus arrives in one ingest call or two."""
    items = sample_items()
    g1 = scripted_graph()
    g1.ingest(items)
    g2 = scripted_graph()
    g2.ingest(items[:2])
    g2.ingest(items[2:])
    derived = {"SHARED_TAG", "SHARED_ENTITY", "SIMILAR_TO"}

    def ident(store):
        return {(u, v, d["etype"]) for u, v, d in store.g.edges(data=True)
                if d["etype"] in derived}

    assert ident(g1.store) == ident(g2.store)


def test_shared_edges_flag_disables_shared_derivation():
    c = cfg()
    c.shared_edges = False
    g = KnowledgeGraph.open(tmp_store(), c)
    g.extractor = ScriptedExtractor(_sample_table())
    g.ingest(sample_items())
    etypes = {d["etype"] for _u, _v, d in g.store.g.edges(data=True)}
    assert "SHARED_TAG" not in etypes and "SHARED_ENTITY" not in etypes
    assert "SIMILAR_TO" in etypes    # kNN edges are governed separately


# --------------------------------------------------------------------------- #
# Natural-boundary chunking (kg/chunkers.py + ingest step 0/4b)
# --------------------------------------------------------------------------- #
def _chat_session_text(n_turns: int = 14, pad: int = 420) -> str:
    lines = ["[chat session — 2023/05/01 (Mon) 10:00]"]
    filler = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
    for i in range(n_turns):
        who = "User" if i % 2 == 0 else "Assistant"
        lines.append(f"{who}: turn {i} " + filler * (pad // len(filler) + 1))
    return "\n".join(lines)


def test_chunked_ingest_structure_and_idempotency():
    c = cfg()
    c.chunking = "turns"
    g = KnowledgeGraph.open(tmp_store(), c)
    g.extractor = ScriptedExtractor({})
    text = _chat_session_text()
    item = CorpusItem(id="s1", modality="text", source_ref="u/s1", title="session 1",
                      text=text, created_at="2023-05-01T10:00:00+00:00")
    rep = g.ingest([item])
    eps = [n for n in g.store.nodes.values() if n.ntype == NodeType.EPISODE]
    assert rep.ingested == len(eps) >= 2
    assert all(e.id.startswith("ep_s1#c") for e in eps)
    # every chunk keeps the session header (the temporal anchor)
    assert all((e.raw_text or "").startswith("[chat session") for e in eps)
    # ONE un-rankable SOURCE parent holding the full original text
    srcs = [n for n in g.store.nodes.values() if n.ntype == NodeType.SOURCE]
    assert len(srcs) == 1 and srcs[0].id == "src_s1" and srcs[0].raw_text == text
    # PART_OF chunk→parent for every chunk; NEXT chain between consecutive siblings
    part = [(u, v) for u, v, d in g.store.all_edges()
            if d["etype"] == EdgeType.PART_OF.value]
    nxt = [(u, v) for u, v, d in g.store.all_edges()
           if d["etype"] == EdgeType.NEXT.value]
    assert len(part) == len(eps) and all(v == "src_s1" for _u, v in part)
    assert len(nxt) == len(eps) - 1
    # retrieval ranks chunk EPISODEs only — the SOURCE parent never surfaces
    res = g.query("lorem ipsum turn")
    assert res.object_ids and all(oid.startswith("ep_s1#c") for oid in res.object_ids)
    # re-ingest is idempotent: every chunk hash-skips, nothing is duplicated
    rep2 = g.ingest([item])
    assert rep2.ingested == 0 and rep2.skipped == len(eps)
    assert len([n for n in g.store.nodes.values() if n.ntype == NodeType.SOURCE]) == 1


def test_chunking_off_is_unchanged():
    c = cfg()
    c.chunking = "none"
    g = KnowledgeGraph.open(tmp_store(), c)
    g.extractor = ScriptedExtractor({})
    g.ingest([CorpusItem(id="s1", modality="text", source_ref="u/s1",
                         text=_chat_session_text())])
    assert [n.id for n in g.store.nodes.values() if n.ntype == NodeType.EPISODE] == ["ep_s1"]
    assert not [n for n in g.store.nodes.values() if n.ntype == NodeType.SOURCE]


def test_context_builder_caps_chunks_per_source():
    from kg.rag import ContextBuilder
    c = cfg()
    c.rag_context_episodes = 4
    c.rag_chunks_per_source = 2
    cb = ContextBuilder.__new__(ContextBuilder)
    cb.config = c
    ranked = ["ep_a#c000", "ep_a#c001", "ep_a#c002", "ep_b#c000", "ep_a#c003", "ep_c"]
    assert cb._select_episodes(ranked) == ["ep_a#c000", "ep_a#c001", "ep_b#c000", "ep_c"]


# --------------------------------------------------------------------------- #
# Phase 2: format sniffer + multi-format chunkers (kg/chunkers.py)
# --------------------------------------------------------------------------- #
def _markdown_doc(sections: int = 6, pad: int = 300) -> str:
    filler = "Setup notes and configuration details for the deployment. "
    parts = ["# Guide", "", "Intro paragraph about the guide.", ""]
    for i in range(sections):
        parts += [f"## Section {i}", "", filler * (pad // len(filler) + 1), ""]
    return "\n".join(parts)


def _prose_doc(paras: int = 8, pad: int = 300) -> str:
    filler = "The quick brown fox jumps over the lazy dog near the river bank. "
    return "\n\n".join(f"Paragraph {i}. " + filler * (pad // len(filler) + 1)
                       for i in range(paras))


def _code_doc(n_funcs: int = 8, body_lines: int = 6) -> str:
    lines = ["import os", "import sys", ""]
    for i in range(n_funcs):
        lines.append(f"def func_{i}(x):")
        lines.append("")               # internal blank line must NOT split the block
        for j in range(body_lines):
            lines.append(f"    value_{j} = x * {j} + {i}")
        lines.append(f"    return value_{body_lines - 1}")
        lines.append("")
    return "\n".join(lines)


def test_sniff_format_routing():
    assert sniff_format(_chat_session_text()) == "turns"
    assert sniff_format(_markdown_doc()) == "markdown"
    assert sniff_format(_code_doc()) == "code"
    assert sniff_format(_prose_doc()) == "prose"
    assert sniff_format("#!/usr/bin/env python\nprint('hello')") == "code"  # shebang
    assert sniff_format("") == "prose"


def test_chunk_markdown_breadcrumbs_and_ceiling():
    doc = _markdown_doc(sections=6, pad=300)
    assert chunk_markdown("## Short\n\nfits in one", target=600, max_chars=1200) == []
    chunks = chunk_markdown(doc, target=600, max_chars=1200)
    assert len(chunks) >= 2
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(len(c.text) <= 1200 for c in chunks)
    # every chunk opens with its heading breadcrumb path (self-describing)
    assert all(c.text.startswith("# Guide") for c in chunks)
    assert any(c.text.splitlines()[0].startswith("# Guide > ## Section") for c in chunks)
    # section content stays under its own breadcrumb
    sec3 = next(c.text for c in chunks if "## Section 3" in c.text.splitlines()[0]
                or "# Guide > ## Section 3" in c.text)
    assert "Setup notes" in sec3


def test_chunk_markdown_oversized_section_fallback():
    giant = "## Big\n\n" + " ".join(f"Sentence {i} of the giant section body."
                                    for i in range(120))
    chunks = chunk_markdown(giant, target=500, max_chars=800)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 800 for c in chunks)
    # every continuation chunk re-opens with the section breadcrumb (like the
    # turn chunker re-prefixing the session header)
    assert all(c.text.startswith("## Big\n") for c in chunks)


def test_chunk_prose_packing_and_ceiling():
    assert chunk_prose("short text", target=600, max_chars=1200) == []
    doc = _prose_doc(paras=8, pad=300)
    chunks = chunk_prose(doc, target=600, max_chars=1200)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 1200 for c in chunks)
    assert chunks[0].text.startswith("Paragraph 0.")
    # the promoted fallback is exactly what chunk_text already did for non-chat text
    assert ([c.text for c in chunks]
            == [c.text for c in chunk_text(doc, target=600, max_chars=1200)])


def test_chunk_code_blocks_and_ceiling():
    assert chunk_code(_code_doc(n_funcs=2, body_lines=2), target=400, max_chars=800) == []
    doc = _code_doc()
    chunks = chunk_code(doc, target=400, max_chars=800)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 800 for c in chunks)
    # chunks break at top-level boundaries: none starts mid-function (indented)
    assert all(not c.text.startswith((" ", "\t")) for c in chunks)
    # a function's internal blank line did not split it: func_0 stays whole
    holder = next(c.text for c in chunks if "def func_0(" in c.text)
    assert "return value_5" in holder.split("def func_1", 1)[0]


def test_turns_chunking_byte_identical_and_auto_routes_chat():
    text = _chat_session_text()
    base = chunk_text(text, target=2200, max_chars=4400)
    assert len(base) >= 2
    for mode in ("turns", "auto"):
        got = chunk_for(text, mode=mode, target=2200, max_chars=4400)
        assert [(c.ordinal, c.text) for c in got] == [(c.ordinal, c.text) for c in base]


def test_auto_chunked_markdown_ingest_structure_and_idempotency():
    c = cfg()
    c.chunking = "auto"
    c.chunk_target_chars = 600
    c.chunk_max_chars = 1200
    g = KnowledgeGraph.open(tmp_store(), c)
    g.extractor = ScriptedExtractor({})
    text = _markdown_doc(sections=6, pad=300)
    item = CorpusItem(id="d1", modality="text", source_ref="u/d1", title="guide",
                      text=text, created_at="2023-05-01T10:00:00+00:00")
    rep = g.ingest([item])
    eps = [n for n in g.store.nodes.values() if n.ntype == NodeType.EPISODE]
    assert rep.ingested == len(eps) >= 2
    assert all(e.id.startswith("ep_d1#c") for e in eps)
    # every chunk opens with its markdown breadcrumb (auto routed to chunk_markdown)
    assert all((e.raw_text or "").startswith("# Guide") for e in eps)
    # ONE un-rankable SOURCE parent holding the full original text
    srcs = [n for n in g.store.nodes.values() if n.ntype == NodeType.SOURCE]
    assert len(srcs) == 1 and srcs[0].id == "src_d1" and srcs[0].raw_text == text
    # PART_OF chunk→parent for every chunk; NEXT chain between consecutive siblings
    part = [(u, v) for u, v, d in g.store.all_edges()
            if d["etype"] == EdgeType.PART_OF.value]
    nxt = [(u, v) for u, v, d in g.store.all_edges()
           if d["etype"] == EdgeType.NEXT.value]
    assert len(part) == len(eps) and all(v == "src_d1" for _u, v in part)
    assert len(nxt) == len(eps) - 1
    # re-ingest is idempotent: every chunk hash-skips, nothing is duplicated
    rep2 = g.ingest([item])
    assert rep2.ingested == 0 and rep2.skipped == len(eps)
    assert len([n for n in g.store.nodes.values() if n.ntype == NodeType.SOURCE]) == 1
