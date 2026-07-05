"""Tests for the graph-RAG answer flow (kg/rag.py).

The query path is retrieve-then-read: PPR (no LLM) builds a context of episodes + valid
facts, then a SINGLE LLM call answers over it. These tests assert the LLM does NOT traverse
(one create() call), citations are validated against the context, and the as-of view reads
the world as it was then.

The answer path is now LIVE-ONLY (the selectable offline answerer was removed). To keep this
suite deterministic and free we inject a FAKE OpenAI client into `g.ask(..., client=...)`,
so `OpenAIAnswerer` runs end-to-end without touching the real API. Extraction is stubbed with
a `ScriptedExtractor`; embeddings use the real local bge model ("st" — deterministic, no key,
no network once cached). Run: python -m pytest tests/test_rag.py -q
"""
from __future__ import annotations

import json
import os
import tempfile
import types
from unittest import mock

import pytest

from kg import Config, KnowledgeGraph
from kg.embedders import SentenceTransformerEmbedder, get_embedder
from kg.extractors import ScriptedExtractor
from kg.rag import (
    ContextBuilder,
    FactLine,
    RagAnswerer,
    _extractive,
    _validate,
    get_answerer,
)
from kg.synthetic import becky_stream


def cfg() -> Config:
    c = Config.default()
    c.embedder = "st"   # real local bge — deterministic, free, no key, no network once cached
    return c


def becky_graph() -> KnowledgeGraph:
    """Open a graph over the synthetic Becky stream with a ScriptedExtractor.

    KnowledgeGraph.__init__ builds an extractor via get_extractor(), which is now live-only
    and would need a key. We patch it to a ScriptedExtractor so this helper works with NO
    OPENAI_API_KEY (and never makes a real extraction call); the embedder is the real bge.
    """
    items, table = becky_stream()
    scripted = ScriptedExtractor(table)
    with mock.patch("kg.graph.get_extractor", return_value=scripted):
        g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), cfg())
    g.extractor = scripted
    g.ingest(items)
    return g


# ---- scripted fake OpenAI client (no real API; OpenAIAnswerer runs over it) -------- #
class _FakeOpenAI:
    """Minimal stand-in for openai.OpenAI. `.chat.completions.create(**kw)` returns a message
    whose single tool call is `submit_answer`. A zeroed `.usage` is attached so
    UsageMeter.record(...) reads real attributes without crashing."""

    def __init__(self, answer: str = "", citations: list[str] | None = None):
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
# openai path via injected fake client — grounding, citations, as-of
# (these replace the old selectable-offline-answerer tests)
# --------------------------------------------------------------------------- #
def test_answer_is_grounded_and_cites_episodes():
    g = becky_graph()
    client = _FakeOpenAI(answer="Becky lives in Berlin and works with Dana.",
                         citations=["ep_becky02", "ep_becky04"])
    ans = g.ask("Where does Becky live and who does she work with?", client=client)
    assert ans.backend == "openai"
    assert "Berlin" in ans.answer
    # the current-view facts that drive a grounded answer are surfaced in the context
    assert any("Berlin" in f for f in ans.facts)
    # citations the model returned are kept because they ARE in the retrieved context
    assert ans.citations and all(c.startswith("ep_") for c in ans.citations)
    assert set(ans.citations) <= set(ans.context_episodes)


def test_answer_respects_as_of():
    g = becky_graph()
    now = g.ask("Where does Becky live?",
                client=_FakeOpenAI(answer="Berlin.", citations=["ep_becky02"]))
    past = g.ask("Where does Becky live?", as_of="2022",
                 client=_FakeOpenAI(answer="Toronto.", citations=["ep_becky01"]))
    # the FACTS context is temporally filtered: current view vs the world as of 2022
    assert any("Berlin" in f for f in now.facts)
    assert any("Toronto" in f for f in past.facts)
    assert past.as_of == "2022"
    assert now.as_of is None


def test_object_ids_feed_recall():
    from kg.evaluate import _recall_at_k
    g = becky_graph()
    ans = g.ask("Becky Berlin", client=_FakeOpenAI(answer="Berlin.", citations=[]))
    assert ans.object_ids  # the PPR ranking is exposed as the eval seam
    assert 0.0 <= _recall_at_k(ans.object_ids, {"ep_becky02"}, 8) <= 1.0


# --------------------------------------------------------------------------- #
# openai path: the LLM does NOT traverse (exactly one create() call)
# --------------------------------------------------------------------------- #
def test_openai_single_call_no_traversal():
    g = becky_graph()
    client = _FakeOpenAI("Becky lives in Berlin.", [])
    ans = g.ask("Where does Becky live?", client=client)
    assert ans.backend == "openai" and "Berlin" in ans.answer
    assert len(client.calls) == 1   # exactly ONE LLM call — no per-hop tool loop
    # the call is given a system prompt + the context as a user message, plus the submit tool
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert client.calls[0]["messages"][1]["role"] == "user"
    assert client.calls[0]["tool_choice"]["function"]["name"] == "submit_answer"


def test_openai_citation_validation():
    g = becky_graph()
    client = _FakeOpenAI("see evidence", ["ep_nope", "entity_0000"])
    ans = g.ask("Becky", client=client)
    # citations not present in the retrieved context are dropped
    assert set(ans.citations) <= set(ans.context_episodes)
    assert "ep_nope" in ans.dropped_citations and "entity_0000" in ans.dropped_citations


def test_validate_unit():
    kept, dropped = _validate(["ep_a", "ep_a", "ep_b", "x"], ["ep_a", "ep_b"])
    assert kept == ["ep_a", "ep_b"] and dropped == ["x"]


# --------------------------------------------------------------------------- #
# internal extractive synthesizer (the surviving crash-guard) — direct coverage
# --------------------------------------------------------------------------- #
def test_extractive_synthesizes_facts_and_episodes():
    g = becky_graph()
    builder = ContextBuilder(g.store, g.config)
    res = g.query("Where does Becky live and who with?", mode="ppr", k=8)
    ep_ids, facts, _blob = builder.build(res)
    out = _extractive(g.store, res.query, ep_ids, facts)
    # the synthesis leads with the current facts (Berlin), then the supporting episodes
    assert "Berlin" in out
    assert "Relevant current facts:" in out
    assert any(eid in out for eid in ep_ids)


def test_extractive_empty_context_is_graceful():
    g = becky_graph()
    out = _extractive(g.store, "anything", [], [])
    assert "No supporting episodes or facts" in out


def test_extractive_used_as_crash_guard_when_client_raises():
    """If the single live call raises mid-run, OpenAIAnswerer degrades to _extractive rather
    than crashing the whole run — so one transient API error never sinks a test run."""
    class _BoomClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            raise RuntimeError("simulated API failure")

    g = becky_graph()
    ans = g.ask("Where does Becky live?", client=_BoomClient())
    assert ans.backend == "openai"
    # the extractive fallback grounds on the same context (Berlin fact + episode citations)
    assert "Berlin" in ans.answer
    assert ans.citations == ans.context_episodes
    assert any("extractive fallback" in n for n in ans.notes)


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
    ans = g.ask("   ", client=_FakeOpenAI(answer="(empty query)", citations=[]))
    assert ans.answer and not ans.citations


# --------------------------------------------------------------------------- #
# live-only backend selection (the offline answerer was removed)
# --------------------------------------------------------------------------- #
def test_get_embedder_is_sentence_transformer():
    """The hashing embedder was removed; get_embedder always returns the local bge model."""
    assert isinstance(get_embedder(cfg()), SentenceTransformerEmbedder)


def test_answerer_without_client_or_key_raises(monkeypatch):
    """No injected client AND no OPENAI_API_KEY -> RuntimeError. There is no offline
    backend to silently degrade to anymore."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    g = becky_graph()
    c = cfg()
    with pytest.raises(RuntimeError):
        get_answerer(g.store, g.embedder, g.canon, c, client=None)
    with pytest.raises(RuntimeError):
        RagAnswerer(g.store, g.embedder, g.canon, c, client=None)
    # and the public entry point surfaces the same error
    with pytest.raises(RuntimeError):
        g.ask("Where does Becky live?")


# --------------------------------------------------------------------------- #
# sibling-chunk expansion (rag_parent_expand — queryside fix 1)
# --------------------------------------------------------------------------- #
def _chunk_node(store, source: str, idx: int, text: str):
    from kg.models import Node, NodeType
    n = Node(id=f"ep_{source}#c{idx:03d}", ntype=NodeType.EPISODE,
             name=f"chunk {idx}", raw_text=text, created_at="2024-01-01")
    store.add_node(n)
    return n.id


def test_parent_expand_off_is_noop():
    """rag_parent_expand=0 (default) must return the selected list unchanged — the
    no-op guarantee that keeps default-config context byte-identical."""
    g = becky_graph()
    builder = ContextBuilder(g.store, g.config)
    selected = ["ep_a#c001", "ep_a#c003"]
    assert builder._expand_siblings(selected) == selected


def test_parent_expand_pulls_in_sibling_window():
    g = becky_graph()
    for i in range(5):
        _chunk_node(g.store, "sess1", i, f"text of chunk {i}")
    g.config.rag_parent_expand = 1
    builder = ContextBuilder(g.store, g.config)
    out = builder._expand_siblings(["ep_sess1#c002"])
    # c001 and c003 (radius 1) pulled in, c000/c004 (radius 2) left out, contiguous order
    assert out == ["ep_sess1#c001", "ep_sess1#c002", "ep_sess1#c003"]


def test_parent_expand_respects_budget_and_keeps_selected():
    g = becky_graph()
    for i in range(5):
        _chunk_node(g.store, "sess1", i, "x" * 100)
    g.config.rag_parent_expand = 2
    g.config.rag_expand_budget_chars = 150   # room for the selected chunk + ~1 sibling
    builder = ContextBuilder(g.store, g.config)
    out = builder._expand_siblings(["ep_sess1#c002"])
    assert "ep_sess1#c002" in out            # originally selected chunk always kept
    assert len(out) < 5                      # budget stopped full expansion


def test_parent_expand_lowest_ranked_source_cut_first():
    """When the budget forces a cut, the best-ranked (first-appearing) source's siblings
    survive; the lower-ranked source's siblings are the ones dropped."""
    g = becky_graph()
    for i in range(3):
        _chunk_node(g.store, "best", i, "x" * 50)
        _chunk_node(g.store, "worst", i, "x" * 50)
    g.config.rag_parent_expand = 1
    # 2 selected chunks (50 each) + budget for exactly one more sibling (50 more)
    g.config.rag_expand_budget_chars = 150
    builder = ContextBuilder(g.store, g.config)
    out = builder._expand_siblings(["ep_best#c001", "ep_worst#c001"])
    assert any(e.startswith("ep_best#c0") and e != "ep_best#c001" for e in out)
    assert not any(e.startswith("ep_worst#c0") and e != "ep_worst#c001" for e in out)


# --------------------------------------------------------------------------- #
# honest fact-date rendering (queryside fix 3a)
# --------------------------------------------------------------------------- #
def test_factline_render_mentioned_vs_since_until():
    open_fact = FactLine(src="Becky", rel="lives_in", dst="Berlin", valid_at="2024-03-05")
    assert "mentioned 2024-03-05" in open_fact.render()
    assert "since" not in open_fact.render()

    closed_fact = FactLine(src="Becky", rel="lived_in", dst="Toronto",
                           valid_at="2022-01-01", invalid_at="2023-06-01")
    rendered = closed_fact.render()
    assert "since 2022-01-01" in rendered and "until 2023-06-01" in rendered


# --------------------------------------------------------------------------- #
# numeric judge_suspect heuristic (queryside fix 5)
# --------------------------------------------------------------------------- #
def test_response_proxy_numeric_reference_requires_asserted_value():
    from kg.testrun import _response_proxy
    # cumulative-vs-increment miscount: answer mentions "25" while asserting 50
    proxy = _response_proxy(
        "You now have 50 new postcards total: it includes 17 from August and 25 from "
        "November.", "25")
    assert proxy["contains"] is False

    proxy_correct = _response_proxy("You've added 25 new postcards since you restarted.",
                                    "25")
    assert proxy_correct["contains"] is True


def test_response_proxy_non_numeric_reference_unchanged():
    from kg.testrun import _response_proxy
    proxy = _response_proxy("I stayed in a hostel in Tokyo.", "hostel in Tokyo")
    assert proxy["contains"] is True
