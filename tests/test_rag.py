"""Tests for the graph-RAG answer flow (kg/rag.py).

The query path is retrieve-then-read: PPR (no LLM) builds a context of episodes + valid
facts, then a SINGLE LLM call answers over it. These tests assert the LLM does NOT traverse
(one create() call), citations are validated against the context, and the offline answerer
gives a grounded, deterministic answer with no API key. Run: python -m pytest -q
"""
from __future__ import annotations

import os
import tempfile
import types

from kg import Config, KnowledgeGraph
from kg.corpus import CorpusItem
from kg.extractors import ScriptedExtractor
from kg.rag import ContextBuilder, OfflineAnswerer, _validate
from kg.synthetic import becky_stream


def cfg() -> Config:
    c = Config.default()
    c.embedder = "hashing"
    c.extractor = "heuristic"
    return c


def becky_graph() -> KnowledgeGraph:
    g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), cfg())
    items, table = becky_stream()
    g.extractor = ScriptedExtractor(table)
    g.ingest(items)
    return g


# ---- scripted fake Anthropic client (mirrors _FakeL3Client) ---------------- #
class _FakeAnswerClient:
    def __init__(self, answer: str, citations: list[str]):
        self.messages = self
        self.calls: list[dict] = []
        self._a, self._c = answer, citations

    def create(self, **kw):
        self.calls.append(kw)
        tu = types.SimpleNamespace(type="tool_use", name="submit_answer",
                                   input={"answer": self._a, "citations": self._c})
        return types.SimpleNamespace(content=[tu])


# --------------------------------------------------------------------------- #
# offline answerer
# --------------------------------------------------------------------------- #
def test_offline_answer_is_grounded_and_cites_episodes():
    g = becky_graph()
    ans = g.ask("Where does Becky live and who does she work with?", backend="offline")
    assert ans.backend == "offline"
    assert ans.citations and all(c.startswith("ep_") for c in ans.citations)
    # current facts drive the answer
    assert any("Berlin" in f for f in ans.facts)
    assert "Berlin" in ans.answer


def test_offline_answer_respects_as_of():
    g = becky_graph()
    now = g.ask("Where does Becky live?", backend="offline")
    past = g.ask("Where does Becky live?", backend="offline", as_of="2022")
    assert any("Berlin" in f for f in now.facts)
    assert any("Toronto" in f for f in past.facts)
    assert past.as_of == "2022"


def test_object_ids_feed_recall():
    from kg.evaluate import _recall_at_k
    g = becky_graph()
    ans = g.ask("Becky Berlin", backend="offline")
    assert ans.object_ids  # the PPR ranking is exposed as the eval seam
    assert 0.0 <= _recall_at_k(ans.object_ids, {"ep_becky02"}, 8) <= 1.0


# --------------------------------------------------------------------------- #
# claude path via injected fake client (the LLM does NOT traverse)
# --------------------------------------------------------------------------- #
def test_claude_single_call_no_traversal():
    g = becky_graph()
    client = _FakeAnswerClient("Becky lives in Berlin.", [])
    ans = g.ask("Where does Becky live?", client=client)
    assert ans.backend == "claude" and "Berlin" in ans.answer
    assert len(client.calls) == 1   # exactly ONE LLM call — no per-hop tool loop
    # the call was given the context as a single user message, plus the submit tool
    assert client.calls[0]["messages"][0]["role"] == "user"
    assert client.calls[0]["tool_choice"]["name"] == "submit_answer"


def test_claude_citation_validation():
    g = becky_graph()
    client = _FakeAnswerClient("see evidence", ["ep_nope", "entity_0000"])
    ans = g.ask("Becky", client=client)
    # citations not present in the retrieved context are dropped
    assert set(ans.citations) <= set(ans.context_episodes)
    assert "ep_nope" in ans.dropped_citations and "entity_0000" in ans.dropped_citations


def test_validate_unit():
    kept, dropped = _validate(["ep_a", "ep_a", "ep_b", "x"], ["ep_a", "ep_b"])
    assert kept == ["ep_a", "ep_b"] and dropped == ["x"]


# --------------------------------------------------------------------------- #
# context builder
# --------------------------------------------------------------------------- #
def test_context_builder_surfaces_current_facts():
    g = becky_graph()
    builder = ContextBuilder(g.store, g.config)
    res = g.query("Where does Becky live and who with?", mode="ppr", k=8)
    ep_ids, facts, blob = builder.build(res)
    rendered = " ".join(f.render() for f in facts)
    assert "Berlin" in rendered and "Toronto" not in rendered  # current view only
    assert "FACTS" in blob and "EPISODES" in blob


def test_empty_query_does_not_crash():
    g = becky_graph()
    ans = g.ask("   ", backend="offline")
    assert ans.answer and not ans.citations


def test_offline_answerer_direct():
    g = becky_graph()
    builder = ContextBuilder(g.store, g.config)
    res = g.query("Becky Berlin", mode="ppr", k=5)
    ans = OfflineAnswerer(g.store, g.config, builder).answer(res)
    assert ans.backend == "offline" and ans.object_ids == res.object_ids
