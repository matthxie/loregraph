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
    _RAG_SYS,
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


def test_invalid_citations_also_stripped_from_answer_text():
    """PROTOCOL §3.12: ids that fail the gate leave the answer TEXT too, not just the
    citations list — the reader must never see invented evidence."""
    g = becky_graph()
    client = _FakeOpenAI("Becky lives in Berlin [ep_nope]. True fact.", ["ep_nope"])
    ans = g.ask("Where does Becky live?", client=client)
    assert "ep_nope" in ans.dropped_citations
    assert "ep_nope" not in ans.answer
    assert "Berlin" in ans.answer                      # only the bad id is removed


def test_strip_citations_unit():
    from kg.rag import strip_citations
    assert strip_citations("A [ep_x]. B", ["ep_x"]) == "A. B"
    assert strip_citations("A [ep_a, ep_x] B", ["ep_x"]) == "A [ep_a] B"
    assert strip_citations("A [ep_x, ep_a] B", ["ep_x"]) == "A [ep_a] B"
    assert strip_citations("bare ep_x end", ["ep_x"]) == "bare end"
    assert strip_citations("chunk [ep_s#c001] here", ["ep_s#c001"]) == "chunk here"
    assert strip_citations("keep [ep_a]", []) == "keep [ep_a]"
    # an id that prefixes another id must not damage the longer one
    assert strip_citations("see [ep_ab]", ["ep_a"]) == "see [ep_ab]"
    # only episode-id-SHAPED dropped strings are removed: a model listing an entity
    # name or a year as a "citation" must not lose that token from valid prose
    assert strip_citations("Sam is at Figma [ep_x]. Figma is a tool.",
                           ["Figma", "ep_x"]) == "Sam is at Figma. Figma is a tool."
    # cleanup is scoped to the removal sites — unrelated formatting survives
    assert strip_citations("a  b   c [ep_x]", ["ep_x"]) == "a  b   c"
    assert strip_citations("keep : this ! [ep_x] gone", ["ep_x"]) == \
        "keep : this ! gone"


def test_search_and_answer_carry_structured_fact_rows():
    """Structured-first (PROTOCOL §3): context facts ride as full Fact objects
    alongside the proven rendered lines, on both the search and answer paths, and
    the answer now carries the §5.1 block it actually read (context_text)."""
    g = becky_graph()
    res = g.search("Where does Becky live?")
    assert res.facts and res.fact_rows
    assert [r["rendered"] for r in res.fact_rows] == res.facts
    row = res.fact_rows[0]
    assert set(row) == {"source", "predicate", "target", "status", "valid_from",
                        "valid_to", "recorded_at", "episode_id", "confidence",
                        "provenance", "functional", "disputed_by", "rendered"}
    assert row["status"] in ("asserted", "ended")
    assert row["episode_id"] and row["episode_id"].startswith("ep_")

    ans = g.ask("Where does Becky live?", client=_FakeOpenAI("Berlin", []))
    assert [r["rendered"] for r in ans.fact_rows] == ans.facts
    assert "EPISODES" in ans.context_text          # the §5.1 block the model read


def test_context_builder_enforces_since_until_window():
    """§7.3: the window is a hard bound on the RETURNED episodes. The context
    builder re-injects episodes after the retriever's filter (sibling expansion,
    provenance promotion), so it must re-apply the window read off the result."""
    from kg.models import NodeType
    from kg.retrieval import RetrievalResult
    g = becky_graph()
    eps = sorted(g.store.nodes_of_type(NodeType.EPISODE),
                 key=lambda n: n.created_at)
    early, late = eps[0], eps[-1]
    assert early.created_at[:10] < late.created_at[:10]
    result = RetrievalResult(query="Becky", mode="ppr",
                             objects=[(early.id, 1.0), (late.id, 0.9)])
    result.window = (late.created_at[:10], None)     # since = the late episode's day
    builder = ContextBuilder(g.store, cfg())
    ctx_ids, _facts, blob = builder.build(result)
    assert early.id not in ctx_ids
    assert early.id not in blob


def test_wire_facts_serves_rows_and_wraps_strings():
    from kg.daemon import Daemon
    rows = Daemon._wire_facts([{"source": "a", "rendered": "r"}, "plain line", 7])
    assert rows == [{"source": "a", "rendered": "r"}, {"rendered": "plain line"}]


def test_sys_prompt_orders_timeframe_answer_first():
    """07741c44: the reader led with the current state and the judge graded the first
    clause, missing the earlier timeframe the question actually asked about."""
    assert "timeframe" in _RAG_SYS.lower()
    assert "state the answer for that timeframe first" in _RAG_SYS.lower()


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


# ---- fake client that scripts a SEQUENCE of responses (retry-on-empty tests) ------- #
class _SequencedFakeOpenAI:
    """Like _FakeOpenAI but returns a different scripted response per call, so tests can
    simulate a first call that comes back empty (finish_reason='length', no tool call —
    e.g. a reasoning model burning its whole completion budget on reasoning tokens) and a
    second (retry) call that succeeds."""

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self.chat = self
        self.completions = self
        self.calls: list[dict] = []

    def create(self, **kw):
        self.calls.append(kw)
        resp = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        if resp.get("empty"):
            message = types.SimpleNamespace(content=None, tool_calls=None)
        else:
            tc = types.SimpleNamespace(
                id="call_0",
                function=types.SimpleNamespace(
                    name="submit_answer",
                    arguments=json.dumps({"answer": resp.get("answer", ""),
                                          "citations": resp.get("citations", [])})))
            message = types.SimpleNamespace(content=None, tool_calls=[tc])
        choice = types.SimpleNamespace(message=message,
                                       finish_reason=resp.get("finish_reason", "tool_calls"))
        usage = types.SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        return types.SimpleNamespace(choices=[choice], usage=usage)


# --------------------------------------------------------------------------- #
# empty-answer retry (reader5-queryside-both-retarget-smoke 06878be2: a reasoning model
# can burn its whole completion budget on reasoning tokens and never emit submit_answer —
# finish_reason="length" with no content/tool call, not an API error)
# --------------------------------------------------------------------------- #
def test_empty_length_response_retries_with_doubled_token_cap():
    g = becky_graph()
    client = _SequencedFakeOpenAI([
        {"empty": True, "finish_reason": "length"},
        {"answer": "Becky lives in Berlin.", "citations": ["ep_becky02"],
         "finish_reason": "tool_calls"},
    ])
    ans = g.ask("Where does Becky live?", client=client)
    assert len(client.calls) == 2                         # exactly one retry
    def _cap(call):    # gpt-5/o-series send max_completion_tokens instead of max_tokens
        return call.get("max_tokens") or call["max_completion_tokens"]
    assert _cap(client.calls[1]) == _cap(client.calls[0]) * 2    # doubled on retry
    assert "Berlin" in ans.answer
    assert any("retried with doubled token cap" in n for n in ans.notes)


def test_empty_length_response_still_empty_after_retry_falls_back():
    g = becky_graph()
    client = _SequencedFakeOpenAI([
        {"empty": True, "finish_reason": "length"},
        {"empty": True, "finish_reason": "length"},
    ])
    ans = g.ask("Where does Becky live?", client=client)
    assert len(client.calls) == 2                          # retried once, not looped forever
    assert any("used extractive fallback" in n for n in ans.notes)
    assert ans.answer and ans.answer != "(no answer produced)"


def test_non_length_empty_response_does_not_retry():
    """An empty answer NOT caused by hitting the length limit (e.g. the model just chose
    not to call the tool) must not trigger the retry — only finish_reason='length' does."""
    g = becky_graph()
    client = _SequencedFakeOpenAI([{"empty": True, "finish_reason": "stop"}])
    ans = g.ask("Where does Becky live?", client=client)
    assert len(client.calls) == 1
    assert ans.answer == "(no answer produced)"


# --------------------------------------------------------------------------- #
# lexical retarget payload bonus (retarget smoke 0bb5a684: the lexical scorer picked a
# question-echoing chunk with no dates over the chunks carrying the actual dates)
# --------------------------------------------------------------------------- #
def test_lex_payload_bonus_beats_question_echo():
    g = becky_graph()
    _chunk_node(g.store, "sess1", 0,
               "I'm preparing for an upcoming meeting with my team about our schedule")
    _chunk_node(g.store, "sess1", 1, "we met on 2024-03-05 to finalize the schedule")
    g.config.rag_retarget = "seed+lex"
    builder = ContextBuilder(g.store, g.config)
    selected = ["ep_sess1#c000"]
    # c000 (question-echo, no payload) wins on embedding seed score alone
    result = _retrieval_result("when was our meeting about the schedule",
                               selected,
                               seed_scores={"ep_sess1#c000": 0.9, "ep_sess1#c001": 0.1})
    out = builder._retarget_chunks(selected, result)
    assert out == ["ep_sess1#c001"]                        # the dated chunk wins the seat


def test_lex_payload_bonus_tie_keeps_incumbent():
    """Two chunks with identical lex scores (overlap + payload) must not swap — ties keep
    the incumbent, only a STRICT improvement displaces it."""
    g = becky_graph()
    _chunk_node(g.store, "sess1", 0, "the schedule meeting happened on 2024-03-05")
    _chunk_node(g.store, "sess1", 1, "the schedule meeting happened on 2024-04-06")
    g.config.rag_retarget = "seed+lex"
    builder = ContextBuilder(g.store, g.config)
    selected = ["ep_sess1#c000"]
    result = _retrieval_result("when was the schedule meeting", selected,
                               seed_scores={"ep_sess1#c000": 0.9, "ep_sess1#c001": 0.1})
    out = builder._retarget_chunks(selected, result)
    assert out == ["ep_sess1#c000"]                        # tie -> incumbent survives


# --------------------------------------------------------------------------- #
# extended payload lexicon (reader5-queryside-both-retarget-1: 3 P->F regressions where
# the evicted evidence carried relative-date phrases with no digits — "a month ago",
# "last week", "exactly two months ago" — that the digit/date-only payload bonus missed)
# --------------------------------------------------------------------------- #
def test_relative_date_phrase_counts_as_payload():
    g = becky_graph()
    builder = ContextBuilder(g.store, g.config)
    for phrase in ("I switched teams last week", "that happened a month ago",
                   "exactly two months ago I moved apartments",
                   "we spoke yesterday", "I recently changed jobs"):
        assert builder._payload_bonus(phrase) > 0, phrase


def test_spelled_out_quantity_counts_as_payload():
    g = becky_graph()
    builder = ContextBuilder(g.store, g.config)
    assert builder._payload_bonus("it took three months to finish") > 0
    assert builder._payload_bonus("I paid twenty dollars for it") > 0


def test_retarget_keeps_relative_date_evidence_over_question_echo():
    """Reproduces the P->F signature: a question-echo chunk with no digits must not beat
    a same-source chunk whose only payload is a relative-date phrase."""
    g = becky_graph()
    _chunk_node(g.store, "sess1", 0,
               "I wanted to ask you about my job situation and how things changed")
    _chunk_node(g.store, "sess1", 1, "I actually started that new job exactly two months ago")
    g.config.rag_retarget = "seed+lex"
    builder = ContextBuilder(g.store, g.config)
    selected = ["ep_sess1#c000"]
    result = _retrieval_result("when did I start my new job", selected,
                               seed_scores={"ep_sess1#c000": 0.9, "ep_sess1#c001": 0.1})
    out = builder._retarget_chunks(selected, result)
    assert out == ["ep_sess1#c001"]


# --------------------------------------------------------------------------- #
# chunk-level retargeting (rag_retarget / rag_provenance_promote — query-side only)
# --------------------------------------------------------------------------- #
def _retrieval_result(query: str, object_ids: list[str], seed_scores: dict | None = None):
    from kg.retrieval import RetrievalResult
    return RetrievalResult(query=query, mode="ppr",
                           objects=[(eid, 1.0) for eid in object_ids],
                           seed_scores=seed_scores or {})


def test_retarget_off_is_noop():
    """rag_retarget='off' must leave _select_episodes' output byte-identical — the
    no-op guarantee that keeps retargeting-free configs unchanged."""
    g = becky_graph()
    g.config.rag_retarget = "off"
    for i in range(4):
        _chunk_node(g.store, "sess1", i, f"text of chunk {i} UCLA university")
    builder = ContextBuilder(g.store, g.config)
    selected = ["ep_sess1#c000", "ep_sess1#c002"]
    result = _retrieval_result("where is UCLA", selected,
                               seed_scores={"ep_sess1#c000": 0.1, "ep_sess1#c003": 0.9})
    assert builder._retarget_chunks(selected, result) == selected


def test_retarget_seed_swaps_by_embedding_rank():
    """rag_retarget='seed': refill a source's slots by embedding seed score instead of
    PPR chunk order — the decisive chunk (c003, high seed score) replaces the weaker
    non-incumbent selected chunk (c002, low seed score); the incumbent (c000) survives."""
    g = becky_graph()
    for i in range(4):
        _chunk_node(g.store, "sess1", i, f"text of chunk {i}")
    g.config.rag_retarget = "seed"
    builder = ContextBuilder(g.store, g.config)
    selected = ["ep_sess1#c000", "ep_sess1#c002"]
    result = _retrieval_result(
        "query", selected,
        seed_scores={"ep_sess1#c000": 0.2, "ep_sess1#c001": 0.1,
                    "ep_sess1#c002": 0.05, "ep_sess1#c003": 0.9})
    out = builder._retarget_chunks(selected, result)
    assert len(out) == len(selected)                 # swaps only, never additions
    assert "ep_sess1#c000" in out                     # incumbent always survives
    assert "ep_sess1#c003" in out                     # best embedding-seed chunk pulled in
    assert "ep_sess1#c002" not in out                 # weakest non-incumbent displaced
    assert builder.last_retargeted and builder.last_retargeted[0]["kind"] == "retarget"


def test_retarget_lexical_swap_beats_seed_pick():
    """rag_retarget='seed+lex': even after the seed pass, a same-source chunk that
    strictly beats a selected one on question-term/digit overlap swaps in — e.g. the
    chunk actually containing the '440 pages' answer beats a higher-seed-score chunk
    that doesn't mention it."""
    g = becky_graph()
    _chunk_node(g.store, "book", 0, "we discussed the club schedule")
    _chunk_node(g.store, "book", 1, "the novel has 440 pages total")
    g.config.rag_retarget = "seed+lex"
    builder = ContextBuilder(g.store, g.config)
    selected = ["ep_book#c000"]
    # c000 wins on embedding seed score despite not containing the numeric answer
    result = _retrieval_result("how many pages",
                               selected,
                               seed_scores={"ep_book#c000": 0.9, "ep_book#c001": 0.1})
    out = builder._retarget_chunks(selected, result)
    assert out == ["ep_book#c001"]


def test_provenance_promote_displaces_expansion_sibling_only():
    """rag_provenance_promote: a fact's source chunk is pulled into context when its
    src/dst overlap the question terms, displacing only a lowest-ranked expansion
    sibling — never an originally selected chunk."""
    g = becky_graph()
    _chunk_node(g.store, "sess1", 0, "selected chunk text")
    _chunk_node(g.store, "sess1", 1, "sibling chunk text")   # expansion sibling
    _chunk_node(g.store, "sess1", 5, "UCLA campus visit details")  # not adjacent -> not pulled by expansion
    g.config.rag_provenance_promote = True
    builder = ContextBuilder(g.store, g.config)
    selected = ["ep_sess1#c000"]
    ctx_ids = ["ep_sess1#c000", "ep_sess1#c001"]   # as if expansion already added the sibling
    facts = [FactLine(src="Becky", rel="visited", dst="UCLA", episode_id="ep_sess1#c005")]
    out = builder._promote_provenance(ctx_ids, selected, facts, "tell me about UCLA")
    assert "ep_sess1#c005" in out                 # promoted in
    assert "ep_sess1#c000" in out                 # originally selected chunk never displaced
    assert "ep_sess1#c001" not in out             # expansion sibling was the one displaced
    assert len(out) == len(ctx_ids)               # displacement only, no growth


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


# --------------------------------------------------------------------------- #
# structured events enumeration in the answer tool (rag_answer_events)
# --------------------------------------------------------------------------- #
class _FakeOpenAIEvents(_FakeOpenAI):
    """Fake client whose submit_answer payload also carries an `events` enumeration."""

    def __init__(self, answer="", citations=None, events=None):
        super().__init__(answer, citations)
        self._e = events or []

    def create(self, **kw):
        self.calls.append(kw)
        tc = types.SimpleNamespace(
            id="call_0",
            function=types.SimpleNamespace(
                name="submit_answer",
                arguments=json.dumps({"events": self._e, "answer": self._a,
                                      "citations": self._c})))
        message = types.SimpleNamespace(content=None, tool_calls=[tc])
        choice = types.SimpleNamespace(message=message, finish_reason="tool_calls")
        usage = types.SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        return types.SimpleNamespace(choices=[choice], usage=usage)


def _sent_tool_required(client) -> list:
    return client.calls[0]["tools"][0]["function"]["parameters"]["required"]


def test_answer_events_off_schema_unchanged():
    g = becky_graph()
    g.config.rag_answer_events = "off"
    client = _FakeOpenAI("Berlin.", [])
    ans = g.ask("Where does Becky live?", client=client)
    assert "events" not in _sent_tool_required(client)
    assert ans.events == []


def test_answer_events_lanes_gates_by_routed_lane():
    g = becky_graph()
    g.config.rag_answer_events = "lanes"   # default lanes: multihop + state

    # an aggregation question routes to the multihop lane -> events schema required
    ev = [{"date": "2023-01-01", "description": "moved to Berlin", "quantity": ""}]
    client = _FakeOpenAIEvents("Berlin.", [], events=ev)
    ans = g.ask("How many times did Becky move?", client=client)
    assert "events" in _sent_tool_required(client)
    assert ans.events == ev
    assert len(client.calls) == 1          # still exactly ONE call — scaffold, not a loop

    # a plain lookup question routes to the single lane -> plain schema
    client2 = _FakeOpenAI("Berlin.", [])
    ans2 = g.ask("Which city does Becky call home?", client=client2)
    assert "events" not in _sent_tool_required(client2)
    assert ans2.events == []


def test_answer_events_all_and_malformed_payload():
    g = becky_graph()
    g.config.rag_answer_events = "all"
    # malformed events (not a list of dicts) must not crash the parse — dicts only survive
    client = _FakeOpenAIEvents("Berlin.", [], events=["not-a-dict", {"date": "d",
                                                                     "description": "x"}])
    ans = g.ask("Where does Becky live?", client=client)
    assert "events" in _sent_tool_required(client)
    assert ans.events == [{"date": "d", "description": "x"}]


# --------------------------------------------------------------------------- #
# In-text relative-date resolution (config.rag_resolve_reldates)
# --------------------------------------------------------------------------- #
def test_annotate_relative_dates_resolves_against_episode_date():
    from kg.rag import _annotate_relative_dates
    # anchor: Monday 2023-03-27
    out = _annotate_relative_dates("I attended the workshop last Saturday.", "2023-03-27")
    assert "last Saturday [= 2023-03-25]" in out
    out = _annotate_relative_dates("yesterday I adopted a cat", "2023-03-27T20:56:00+00:00")
    assert "yesterday [= 2023-03-26]" in out
    out = _annotate_relative_dates("I booked it two months ago.", "2023-03-27")
    assert "two months ago [≈ 2023-01-26]" in out
    out = _annotate_relative_dates("we met a couple of weeks ago", "2023-03-27")
    assert "a couple of weeks ago [≈ 2023-03-13]" in out
    out = _annotate_relative_dates("I started last week and loved it", "2023-03-27")
    assert "last week [≈ 2023-03-20]" in out
    out = _annotate_relative_dates("it happened 3 days ago", "2023-03-27")
    assert "3 days ago [= 2023-03-24]" in out


def test_annotate_relative_dates_is_noop_without_anchor_or_phrase():
    from kg.rag import _annotate_relative_dates
    assert _annotate_relative_dates("last week was great", None) == "last week was great"
    assert _annotate_relative_dates("last week was great", "not-a-date") == "last week was great"
    assert _annotate_relative_dates("nothing relative here", "2023-03-27") == \
        "nothing relative here"


def test_resolve_reldates_off_is_byte_identical_and_on_annotates():
    g = becky_graph()
    assert g.config.rag_resolve_reldates is False
    # inject a relative phrase into an episode and rebuild the context both ways
    from kg.rag import ContextBuilder
    from kg.retrieval import HybridRetriever
    retr = HybridRetriever(g.store, g.embedder, g.canon, g.config)
    result = retr.retrieve("Where does Becky live?", k=g.config.top_k)
    eid = result.object_ids[0]
    node = g.store.get_node(eid)
    node.raw_text = (node.raw_text or node.name) + " She moved last week."
    node.created_at = node.created_at or "2023-03-27T00:00:00+00:00"
    off = ContextBuilder(g.store, g.config).build(result)[2]
    g.config.rag_resolve_reldates = True
    on = ContextBuilder(g.store, g.config).build(result)[2]
    assert "last week [≈" not in off
    assert "last week [≈" in on


# --------------------------------------------------------------------------- #
# search() — retrieval + context assembly, NO answering LLM
# --------------------------------------------------------------------------- #
def test_search_returns_hits_without_llm_or_key():
    """g.search() is ask() minus the answer call: it must work with no OPENAI_API_KEY,
    make zero API calls, and return structured scored hits whose evidence matches the
    context blob ask()'s LLM would see."""
    g = becky_graph()
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)   # prove the path is key-free
        res = g.search("Where does Becky live?")
    assert res.query == "Where does Becky live?"
    assert res.hits, "expected at least one retrieved memory"
    top = res.hits[0]
    assert top.episode_id and top.text
    assert isinstance(top.score, float)
    # every hit appears in the prompt blob (same evidence the answerer would read)
    for h in res.hits:
        assert f"[{h.episode_id}]" in res.context
    assert "QUESTION: Where does Becky live?" in res.context
    # facts are pre-rendered strings, feed-ready
    assert all(isinstance(f, str) for f in res.facts)


def test_search_empty_query_and_caching():
    g = becky_graph()
    res = g.search("   ")
    assert res.hits == [] and res.context == ""
    g.search("Becky")
    first = g._searcher
    g.search("Becky again")
    assert g._searcher is first, "Searcher (and its warm caches) should be reused"
