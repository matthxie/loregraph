"""Blended natural-language search (search_nl): the hybrid walk and raw BM25 run
side by side over the same query and merge by reciprocal-rank fusion, each hit
tagged with the signals that found it. Fully offline — no LLM, no API key."""
from __future__ import annotations

import os
import tempfile
from unittest import mock

import pytest

from kg.config import Config
from kg.engine import Engine, NoteInput
from kg.errors import InvalidInput
from kg.extractors import ScriptedExtractor
from kg.graph import KnowledgeGraph
from kg.retrieval import rrf_fuse
from kg.synthetic import becky_stream


def cfg() -> Config:
    c = Config.default()
    c.embedder = "st"   # real local bge — deterministic, free, no key
    return c


def becky_graph() -> KnowledgeGraph:
    items, table = becky_stream()
    scripted = ScriptedExtractor(table)
    with mock.patch("kg.graph.get_extractor", return_value=scripted):
        g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), cfg())
    g.extractor = scripted
    g.ingest(items)
    return g


# --------------------------------------------------------------------------- #
# rrf_fuse unit
# --------------------------------------------------------------------------- #
def test_rrf_fuse_sums_signals_and_tags_sources():
    fused = rrf_fuse({"semantic": ["a", "b", "c"], "keyword": ["b", "d"]}, k=10)
    rows = {oid: (score, srcs) for oid, score, srcs in fused}
    # b: rank 2 semantic + rank 1 keyword — agreement sums, so it outranks a (rank 1 alone)
    assert rows["b"][0] == pytest.approx(1 / 62 + 1 / 61)
    assert rows["b"][1] == ("semantic", "keyword")
    assert rows["a"] == (pytest.approx(1 / 61), ("semantic",))
    assert rows["d"] == (pytest.approx(1 / 62), ("keyword",))
    assert [oid for oid, _s, _src in fused] == ["b", "a", "d", "c"]


def test_rrf_fuse_deterministic_ties_and_topk():
    # same single-list rank → identical score → id-ordered tie-break
    fused = rrf_fuse({"semantic": ["z"], "keyword": ["a"]}, k=10)
    assert [oid for oid, _s, _src in fused] == ["a", "z"]
    assert len(rrf_fuse({"semantic": list("abcdef"), "keyword": []}, k=3)) == 3


# --------------------------------------------------------------------------- #
# facade: KnowledgeGraph.search_nl
# --------------------------------------------------------------------------- #
def test_search_nl_blends_walk_and_keyword():
    g = becky_graph()
    fused = g.search_nl("Where does Becky live?", k=8)
    assert fused
    # every fused id came from one of the two rankings, sources say which
    walk_ids = {h.episode_id for h in g.search("Where does Becky live?",
                                               k=8, rerank=True).hits}
    kw_ids = {eid for eid, _ in g.keyword_search("Where does Becky live?", k=8)}
    for eid, score, srcs in fused:
        assert eid in walk_ids | kw_ids
        assert set(srcs) <= {"semantic", "keyword"} and srcs
        assert score > 0
    # scores are descending (rank-fused, deterministic)
    scores = [s for _e, s, _src in fused]
    assert scores == sorted(scores, reverse=True)
    # a name query hits both signals somewhere in the list
    assert any(set(srcs) == {"semantic", "keyword"} for _e, _s, srcs in fused)


def test_search_nl_keeps_keyword_top_hit():
    """The point of fusing instead of walk-only: the lexical #1 for an exact name
    must survive into the blended page."""
    g = becky_graph()
    kw = g.keyword_search("Becky", k=8)
    assert kw
    fused_ids = [eid for eid, _s, _src in g.search_nl("Becky", k=8)]
    assert kw[0][0] in fused_ids


def test_search_nl_empty_query_is_empty():
    g = becky_graph()
    assert g.search_nl("", k=5) == []
    assert g.search_nl("   ", k=5) == []


# --------------------------------------------------------------------------- #
# engine wire shape
# --------------------------------------------------------------------------- #
def test_engine_search_nl_wire_shape():
    eng = Engine.open(tempfile.mkdtemp(), {"kind": "mock"})
    eng._g.extractor = ScriptedExtractor({})
    eng.ingest(NoteInput(text="booked the zanzibar diving trip for october",
                         created_at="2026-07-01T00:00:00Z"))
    eng.ingest(NoteInput(text="dentist moved my appointment to thursday",
                         created_at="2026-07-02T00:00:00Z"))
    out = eng.search_nl("that diving vacation I planned", k=5)
    assert out["query"] == "that diving vacation I planned"
    assert out["episodes"]
    top = out["episodes"][0]
    assert set(top) == {"id", "score", "when", "text", "title", "description",
                        "sources", "source_id", "line_span", "source_ref"}
    assert isinstance(top["sources"], list) and top["sources"]
    assert "zanzibar" in top["text"]
    with pytest.raises(InvalidInput):
        eng.search_nl("   ")
    eng.close()
