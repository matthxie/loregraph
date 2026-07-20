"""Tests for answer-time deterministic aggregation (kg/rag.py, docs/OFFLINE_EVAL.md
Round 6a) — the two default-off, query-side knobs `agg_reconcile` and `agg_map_reduce`.

Both move count/sum arithmetic out of the reader and into CODE: reconcile audits the
reader's OWN enumerated events[] after the fact; map-reduce enumerates per source session
up front, merges/dedups/counts in Python, and lets a final reduce call phrase the result.
Fully offline — a scripted fake OpenAI client drives OpenAIAnswerer end to end (no key, no
network beyond the cached local bge embedder). Run: python -m pytest tests/test_agg.py -q
"""
from __future__ import annotations

import json
import os
import tempfile
import types
from unittest import mock

from kg import Config, KnowledgeGraph
from kg.extractors import ScriptedExtractor
from kg.rag import (
    ContextBuilder,
    OpenAIAnswerer,
    _dedup_events,
    _sum_events,
)
from kg.route import route
from kg.synthetic import becky_stream


def cfg(**over) -> Config:
    c = Config.default()
    c.embedder = "st"
    for k, v in over.items():
        setattr(c, k, v)
    return c


def becky_graph(config: Config) -> KnowledgeGraph:
    items, table = becky_stream()
    scripted = ScriptedExtractor(table)
    with mock.patch("kg.graph.get_extractor", return_value=scripted):
        g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), config)
    g.extractor = scripted
    g.ingest(items)
    return g


# --------------------------------------------------------------------------- #
# Fake OpenAI client that dispatches on the forced tool: `list_items` (MAP) returns
# items keyed by which source id appears in the user message; `submit_answer` (REDUCE or
# the plain answer call) returns a scripted answer + citations + events.
# --------------------------------------------------------------------------- #
class _AggFake:
    def __init__(self, *, map_items=None, default_items=None, answer="",
                 citations=None, events=None):
        self.map_items = map_items or {}       # source_id -> [item]
        self.default_items = default_items      # returned when no source key matches
        self._a = answer
        self._c = citations or []
        self._e = events or []
        self.chat = self
        self.completions = self
        self.calls: list[dict] = []

    def create(self, **kw):
        self.calls.append(kw)
        tool = kw["tool_choice"]["function"]["name"]
        user = kw["messages"][-1]["content"]
        if tool == "list_items":
            items = self.default_items if self.default_items is not None else []
            for src, its in self.map_items.items():
                if src in user:
                    items = its
                    break
            payload, fn = {"items": items}, "list_items"
        else:
            payload = {"answer": self._a, "citations": self._c, "events": self._e}
            fn = "submit_answer"
        tc = types.SimpleNamespace(
            id="c0", function=types.SimpleNamespace(name=fn, arguments=json.dumps(payload)))
        message = types.SimpleNamespace(content=None, tool_calls=[tc])
        choice = types.SimpleNamespace(message=message, finish_reason="tool_calls")
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        return types.SimpleNamespace(choices=[choice], usage=usage)

    def map_calls(self):
        return [c for c in self.calls if c["tool_choice"]["function"]["name"] == "list_items"]

    def answer_calls(self):
        return [c for c in self.calls if c["tool_choice"]["function"]["name"] == "submit_answer"]


def _answerer(g: KnowledgeGraph, config: Config, client) -> OpenAIAnswerer:
    return OpenAIAnswerer(g.store, config, ContextBuilder(g.store, config), client=client)


def _result(query, ctx_ids, as_of=None, lane="multihop"):
    return types.SimpleNamespace(query=query, as_of=as_of, lane=lane, object_ids=ctx_ids)


# --------------------------------------------------------------------------- #
# Pure CODE stage — dedup / sum
# --------------------------------------------------------------------------- #
def test_dedup_by_date_and_normalized_description():
    events = [
        {"date": "2024-03-10", "description": "gym session"},
        {"date": "2024-03-10", "description": "the Gym sessions"},   # same date + norm tokens
        {"date": "2024-04-14", "description": "gym session"},        # distinct date
    ]
    merged = _dedup_events(events)
    assert len(merged) == 2


def test_sum_events_parses_amounts_from_fields_and_descriptions():
    events = [
        {"date": "2024-01-01", "description": "car cover", "amount": "$120"},
        {"date": "2024-01-02", "description": "detailing spray for $20 total"},
        {"date": "2024-01-03", "description": "no amount here"},
    ]
    total, parts = _sum_events(events)
    assert total == 140.0
    assert len(parts) == 2   # the amount-less event contributes nothing


# --------------------------------------------------------------------------- #
# agg_reconcile — audit the reader's own enumeration
# --------------------------------------------------------------------------- #
def test_reconcile_corrects_count_mismatch():
    config = cfg(agg_reconcile=True)
    g = becky_graph(config)
    ans = _answerer(g, config, _AggFake())
    events = [{"date": "2025-01-01", "description": "park visit"},
              {"date": "2025-02-01", "description": "museum visit"}]
    out, note = ans._reconcile_answer("Becky went out 5 times.", events,
                                      "How many times did Becky go out?")
    assert "Correction" in out and "2 matching item" in out
    assert note and "5 != 2" in note


def test_reconcile_leaves_matching_count_untouched():
    config = cfg(agg_reconcile=True)
    g = becky_graph(config)
    ans = _answerer(g, config, _AggFake())
    events = [{"date": "2025-01-01", "description": "park visit"},
              {"date": "2025-02-01", "description": "museum visit"}]
    text = "Becky went out 2 times."
    out, note = ans._reconcile_answer(text, events, "How many times did Becky go out?")
    assert out == text and note is None


def test_reconcile_dedups_before_counting():
    config = cfg(agg_reconcile=True)
    g = becky_graph(config)
    ans = _answerer(g, config, _AggFake())
    # the same real occurrence re-mentioned in a later session must count once
    events = [{"date": "2025-01-01", "description": "park visit"},
              {"date": "2025-01-01", "description": "the park visit"},
              {"date": "2025-02-01", "description": "museum visit"}]
    out, _ = ans._reconcile_answer("Becky went out 3 times.", events,
                                   "How many times did Becky go out?")
    assert "2 matching item" in out    # deduped to 2, correcting the stated 3


def test_reconcile_corrects_sum_mismatch():
    config = cfg(agg_reconcile=True)
    g = becky_graph(config)
    ans = _answerer(g, config, _AggFake())
    events = [{"date": "2024-01-01", "description": "car cover", "quantity": "$120"},
              {"date": "2024-01-02", "description": "spray", "quantity": "$20"}]
    out, note = ans._reconcile_answer("You spent $200 in total.", events,
                                      "How much did I spend in total?")
    assert "$140" in out and "Correction" in out
    assert note and "!= 140" in note


def test_reconcile_end_to_end_appends_correction():
    config = cfg(agg_reconcile=True)
    g = becky_graph(config)
    query = "How many times did Becky visit places?"
    assert route(query) == "multihop"        # aggregate → multihop → events schema on
    fake = _AggFake(answer="Becky visited 5 times.",
                    citations=[],
                    events=[{"date": "2025-01-01", "description": "park"},
                            {"date": "2025-02-01", "description": "museum"},
                            {"date": "2025-03-01", "description": "gallery"}])
    ans = g.ask(query, client=fake)
    assert "Correction" in ans.answer and "3 matching item" in ans.answer
    assert any("agg_reconcile" in n for n in ans.notes)


def test_reconcile_skips_date_arithmetic():
    """"How many weeks/months …" matches is_aggregate_question but is a date difference,
    not an occurrence count — reconcile must not miscount it (route sends it to STATE)."""
    config = cfg(agg_reconcile=True)
    g = becky_graph(config)
    ans = _answerer(g, config, _AggFake())
    events = [{"date": "2025-01-01", "description": "flu recovery"},
              {"date": "2025-04-01", "description": "10th jog"}]
    text = "About 15 weeks had passed."
    out, note = ans._reconcile_answer(
        text, events, "How many weeks had passed since I recovered from the flu?")
    assert out == text and note is None


def test_reconcile_leaves_non_aggregate_untouched():
    config = cfg(agg_reconcile=True)
    g = becky_graph(config)
    fake = _AggFake(answer="Becky lives in Berlin.", citations=[],
                    events=[{"date": "x", "description": "y"}])
    ans = g.ask("Where does Becky live?", client=fake)
    assert ans.answer == "Becky lives in Berlin."
    assert not any("agg_reconcile" in n for n in ans.notes)


# --------------------------------------------------------------------------- #
# agg_map_reduce — MAP per session + CODE merge + REDUCE
# --------------------------------------------------------------------------- #
def test_map_fires_once_per_source_with_schema():
    config = cfg(agg_map_reduce=True)
    g = becky_graph(config)
    ctx = ["ep_becky01", "ep_becky02", "ep_becky03"]   # three distinct sources
    fake = _AggFake(map_items={
        "ep_becky01": [{"date": "2021-01-01", "description": "park visit",
                        "verbatim_quote": "went to the park"}],
        "ep_becky02": [{"date": "2021-02-01", "description": "museum visit",
                        "verbatim_quote": "went to the museum"}],
        "ep_becky03": [],
    })
    ans = _answerer(g, config, fake)
    addendum, note = ans._agg_map_reduce(
        _result("How many places did Becky visit?", ctx), ctx, "gpt-4o-mini")
    assert len(fake.map_calls()) == 3                       # one per source session
    for c in fake.map_calls():
        assert c["tools"][0]["function"]["name"] == "list_items"
    assert "Computed count of enumerated items: 2" in addendum
    assert "2 deduped item" in note


def test_map_dedups_rementioned_item():
    config = cfg(agg_map_reduce=True)
    g = becky_graph(config)
    ctx = ["ep_becky01", "ep_becky02"]
    same = {"date": "2021-01-01", "description": "park visit",
            "verbatim_quote": "went to the park"}
    fake = _AggFake(map_items={"ep_becky01": [same],
                               "ep_becky02": [dict(same, description="the park visits")]})
    ans = _answerer(g, config, fake)
    addendum, _ = ans._agg_map_reduce(
        _result("How many times did Becky go to the park?", ctx), ctx, "gpt-4o-mini")
    assert "Computed count of enumerated items: 1" in addendum


def test_map_sum_matches_python():
    config = cfg(agg_map_reduce=True)
    g = becky_graph(config)
    ctx = ["ep_becky01", "ep_becky02"]
    fake = _AggFake(map_items={
        "ep_becky01": [{"date": "2024-01-01", "description": "car cover", "amount": "$120",
                        "verbatim_quote": "the cover was $120"}],
        "ep_becky02": [{"date": "2024-01-02", "description": "spray", "amount": "$20",
                        "verbatim_quote": "spray cost $20"}],
    })
    ans = _answerer(g, config, fake)
    addendum, _ = ans._agg_map_reduce(
        _result("How much did Becky spend in total?", ctx), ctx, "gpt-4o-mini")
    assert "Computed sum over the enumerated items: $140" in addendum


def test_map_empty_abstention_path():
    config = cfg(agg_map_reduce=True)
    g = becky_graph(config)
    ctx = ["ep_becky01", "ep_becky02"]
    fake = _AggFake(map_items={"ep_becky01": [], "ep_becky02": []})
    ans = _answerer(g, config, fake)
    addendum, note = ans._agg_map_reduce(
        _result("How many times did Becky travel?", ctx), ctx, "gpt-4o-mini")
    # abstention-safe: no fabricated "Computed count", explicit insufficiency escape hatch
    assert "Computed count" not in addendum
    assert "no matching items" in addendum and "insufficient" in addendum
    assert "0 items" in note


def test_map_reduce_end_to_end_feeds_reduce_context():
    config = cfg(agg_map_reduce=True)
    g = becky_graph(config)
    query = "How many places did Becky visit?"
    assert route(query) == "multihop"
    fake = _AggFake(default_items=[{"date": "2021-01-01", "description": "a visit",
                                    "verbatim_quote": "visited"}],
                    answer="Becky visited several places.", citations=[])
    ans = g.ask(query, client=fake)
    assert fake.map_calls()                                 # MAP ran
    reduce_call = fake.answer_calls()[-1]                   # REDUCE = final submit_answer
    assert "COMPUTED AGGREGATION" in reduce_call["messages"][1]["content"]
    assert ans.answer == "Becky visited several places."
    assert any("agg_map_reduce" in n for n in ans.notes)


# --------------------------------------------------------------------------- #
# Both knobs OFF — byte-identical answer flow
# --------------------------------------------------------------------------- #
def test_both_knobs_off_is_byte_identical():
    config = cfg()   # agg_reconcile / agg_map_reduce default False
    assert config.agg_reconcile is False and config.agg_map_reduce is False
    g = becky_graph(config)
    query = "How many times did Becky visit places?"      # multihop + aggregate
    fake = _AggFake(answer="Becky visited 5 times.", citations=[],
                    events=[{"date": "2025-01-01", "description": "park"}])
    ans = g.ask(query, client=fake)
    # exactly one LLM call (no MAP calls fired), and the reader saw the plain context blob
    assert len(fake.calls) == 1
    assert fake.calls[0]["messages"][1]["content"] == ans.context_text
    # the answer is the model's verbatim output — no reconciliation correction appended
    assert ans.answer == "Becky visited 5 times."
    assert not any("agg_" in n for n in ans.notes)
