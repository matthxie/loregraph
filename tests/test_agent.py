"""chat.agent contract tests (PROTOCOL §9.2/§9.3) — kg/agent.py loop and the
Engine.agent facade. The daemon-registration halves (probe safety, BUSY, progress
notifications, param validation) live with the app-owned daemon in the brainbrain repo.

Fully offline: the provider LLM is a scripted OpenAI-SDK-shaped client; embeddings use
the real local bge model, same policy as the rest of the suite.
Run: python -m pytest tests/test_agent.py -q
"""
from __future__ import annotations

import json
import os
import tempfile
import types

import pytest

from kg.agent import MAX_STEPS, run_agent
from kg.engine import Engine, NoteInput
from kg.errors import InvalidInput, ProviderUnavailable


def _tc(name: str, args: dict, cid: str = "call_0"):
    return types.SimpleNamespace(
        id=cid, function=types.SimpleNamespace(name=name,
                                               arguments=json.dumps(args)))


def _resp(content=None, tool_calls=None):
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = types.SimpleNamespace(message=message, finish_reason="tool_calls"
                                   if tool_calls else "stop")
    usage = types.SimpleNamespace(prompt_tokens=0, completion_tokens=0)
    return types.SimpleNamespace(choices=[choice], usage=usage)


class _Scripted:
    """OpenAI-SDK-shaped client that replays a fixed sequence of responses; records
    every create() kwargs for loop-shape assertions."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.chat = self
        self.completions = self
        self.calls: list[dict] = []

    def create(self, **kw):
        self.calls.append(kw)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


@pytest.fixture(scope="module")
def eng():
    e = Engine.open(tempfile.mkdtemp(), {"kind": "mock"})
    e.ingest(NoteInput(text="Met Becky at the climbing gym in Toronto.",
                       created_at="2026-07-01T10:00:00Z"))
    e.ingest(NoteInput(text="Planning the Berlin move with Sam.",
                       created_at="2026-07-02T10:00:00Z"))
    # Module-scoped setup runs BEFORE conftest's function-scoped env snapshot: unpin
    # the provider env Engine.open wrote, or "mock" poisons the snapshot and leaks
    # into every test that runs after this module (see conftest docstring).
    os.environ.pop("KG_LLM", None)
    yield e
    e.close()


def _becky_ep(eng) -> str:
    return next(ep["id"] for ep in eng.episodes_list()["episodes"]
                if "Becky" in (ep["text"] or ""))


# --------------------------------------------------------------------------- #
# the loop: tools dispatch to facade verbs, trace, widened citation gate
# --------------------------------------------------------------------------- #
def test_tool_loop_widened_gate_and_trace(eng):
    eid = _becky_ep(eng)
    client = _Scripted([
        _resp(tool_calls=[_tc("graph_search", {"terms": "Becky"})]),
        _resp(tool_calls=[_tc("submit_answer", {
            "answer": f"At the climbing gym [{eid}]. Invented [ep_fake123].",
            "citations": [eid, "ep_fake123"]})]),
    ])
    notes = []
    r = run_agent(eng, "where did I meet Becky?", client=client,
                  provider_kind="openai", progress=lambda n: notes.append(n))
    # widened gate: the search hit's id is citable; the invented id is dropped AND
    # stripped from the answer text (§9.2 → §3.12)
    assert r["citations"] == [eid]
    assert r["invalid_citations"] == ["ep_fake123"]
    assert eid in r["answer"] and "ep_fake123" not in r["answer"]
    assert r["steps"] == 1
    assert r["trace"] == [{"seq": 1, "tool": "graph_search",
                           "input_summary": '{"terms": "Becky"}',
                           "output_summary": "1 episode(s)"}]
    assert r["context"]["episodes"] == [eid]
    assert notes == [{"state": "tool", "tool": "graph_search",
                      "detail": '{"terms": "Becky"}'}]
    # multi-turn transcript: assistant tool_calls + role:"tool" result went back
    final_messages = client.calls[-1]["messages"]
    assert any(m.get("role") == "tool" for m in final_messages)
    assert any(m.get("role") == "assistant" and m.get("tool_calls")
               for m in final_messages)


def test_every_tool_dispatches_to_a_facade_verb(eng):
    """§9.2 tool table: each graph tool answers from the SAME engine verb the daemon
    serves — and every episode id a tool showed joins the citation universe."""
    eid = _becky_ep(eng)
    client = _Scripted([
        _resp(tool_calls=[_tc("graph_retrieve", {"query": "Becky"})]),
        _resp(tool_calls=[_tc("graph_facts", {"entity": "Becky"})]),
        _resp(tool_calls=[_tc("graph_neighbors", {"id": eid})]),
        _resp(tool_calls=[_tc("graph_episode", {"id": eid})]),
        _resp(tool_calls=[_tc("submit_answer",
                              {"answer": f"ok [{eid}]", "citations": [eid]})]),
    ])
    r = run_agent(eng, "who is Becky?", client=client, provider_kind="openai")
    assert [t["tool"] for t in r["trace"]] == [
        "graph_retrieve", "graph_facts", "graph_neighbors", "graph_episode"]
    assert r["steps"] == 4
    assert r["citations"] == [eid]
    assert eid in r["context"]["episodes"]


def test_bad_tool_input_is_an_error_result_not_a_crash(eng):
    client = _Scripted([
        _resp(tool_calls=[_tc("graph_neighbors", {"id": "nope_123"})]),
        _resp(tool_calls=[_tc("graph_episode", {"id": "ep_missing"})]),
        _resp(tool_calls=[_tc("submit_answer",
                              {"answer": "nothing found", "citations": []})]),
    ])
    r = run_agent(eng, "?", client=client, provider_kind="openai")
    assert r["steps"] == 2                       # both bad calls still traced
    assert r["answer"] == "nothing found"
    assert r["trace"][0]["output_summary"] == "error"
    assert r["trace"][1]["output_summary"] == "not found"


def test_max_steps_clamped_and_final_answer_forced(eng):
    """A model that never stops calling tools is cut off: the loop clamps max_steps
    (the daemon's ceiling, §9.2) and forces submit_answer with the tools narrowed to
    it alone."""
    looping = _Scripted([_resp(tool_calls=[_tc("graph_search", {"terms": "x"})])])
    # the scripted client repeats its last response forever; patch a final answer in
    # once the loop forces submit_answer
    orig_create = looping.create

    def create(**kw):
        choice = kw.get("tool_choice")
        if isinstance(choice, dict) and \
                choice.get("function", {}).get("name") == "submit_answer":
            assert [t["function"]["name"] for t in kw["tools"]] == ["submit_answer"]
            return _resp(tool_calls=[_tc("submit_answer",
                                         {"answer": "forced", "citations": []})])
        return orig_create(**kw)

    looping.create = create
    r = run_agent(eng, "?", client=looping, provider_kind="openai", max_steps=3)
    assert r["steps"] == 3                       # clamped: exactly max_steps tool calls
    assert r["answer"] == "forced"
    r = run_agent(eng, "?", client=looping, provider_kind="openai", max_steps=999)
    assert r["steps"] == MAX_STEPS               # daemon-side ceiling


def test_fabricated_ids_in_fact_text_do_not_enter_the_universe():
    """A hostile note can mint an entity whose NAME lexes like an episode id; when a
    fact renders it, the scraped id must not become citable evidence (§9.2: the
    answerer may not invent evidence — nor may a note invent it for them)."""
    from kg.agent import _Evidence
    ev = _Evidence(lambda eid: eid == "ep_real")
    ev.add_fact("ep_fake_evil --related_to--> Alice (mentioned 2026-01-01) [ep_real]")
    assert ev.episode_ids == ["ep_real"]


def test_max_steps_zero_forces_an_immediate_answer(eng):
    calls = []

    class _Forced:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            calls.append(kw)
            assert kw["tool_choice"]["function"]["name"] == "submit_answer"
            return _resp(tool_calls=[_tc("submit_answer",
                                         {"answer": "direct", "citations": []})])

    r = run_agent(eng, "?", client=_Forced(), provider_kind="openai", max_steps=0)
    assert r["answer"] == "direct" and r["steps"] == 0 and len(calls) == 1


def test_malformed_tool_arguments_surface_as_tool_errors(eng):
    """A provider emitting k='a few' must not kill the run (§9.2: the model can
    correct on retry) — the bad scalar is coerced/errored tool-side."""
    eid = _becky_ep(eng)
    client = _Scripted([
        _resp(tool_calls=[_tc("graph_search", {"terms": "Becky", "k": "a few"})]),
        _resp(tool_calls=[_tc("submit_answer",
                              {"answer": f"ok [{eid}]", "citations": [eid]})]),
    ])
    r = run_agent(eng, "?", client=client, provider_kind="openai")
    assert r["citations"] == [eid]               # run survived and found evidence
    assert r["trace"][0]["output_summary"] == "1 episode(s)"


def test_reduced_forced_turn_never_returns_a_tool_json_blob(eng):
    """A CLI model that ignores the 'no tool calls left' instruction must not have
    its raw {"tool": ...} JSON served as the user-visible answer."""

    class _Stubborn:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            return _resp(content=json.dumps(
                {"tool": "graph_search", "arguments": {"terms": "x"}}))

    r = run_agent(eng, "?", client=_Stubborn(), provider_kind="codex", max_steps=1)
    assert r["answer"] == "(no answer produced)"
    assert '"tool"' not in r["answer"]
    assert r["steps"] == 1                       # the one allowed call still ran


def test_reduced_loop_for_cli_shims(eng):
    """§9.4: tool-less CLI shims run the re-prompting loop — one JSON object per
    turn, transcript replayed as text."""
    eid = _becky_ep(eng)

    class _CLI:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.prompts: list[str] = []

        def create(self, **kw):
            self.prompts.append(kw["messages"][0]["content"])
            if len(self.prompts) == 1:
                return _resp(content=json.dumps(
                    {"tool": "graph_search", "arguments": {"terms": "Becky"}}))
            return _resp(content=json.dumps(
                {"answer": f"gym [{eid}] but also [ep_bogus]",
                 "citations": [eid, "ep_bogus"]}))

    cli = _CLI()
    r = run_agent(eng, "where did I meet Becky?", client=cli,
                  provider_kind="codex")
    assert r["steps"] == 1 and r["trace"][0]["tool"] == "graph_search"
    assert r["citations"] == [eid] and r["invalid_citations"] == ["ep_bogus"]
    assert "ep_bogus" not in r["answer"]
    assert "TOOL CALL 1" in cli.prompts[1]       # transcript replayed as text


# --------------------------------------------------------------------------- #
# Engine.agent facade — provider taxonomy identical to answer()
# --------------------------------------------------------------------------- #
def test_engine_agent_mock_and_validation(eng):
    r = eng.agent("where did I meet Becky?")
    assert "mock provider" in r["answer"]        # canned submit_answer, zero steps
    assert r["steps"] == 0 and r["trace"] == []
    with pytest.raises(InvalidInput):
        eng.agent("   ")


def test_engine_agent_provider_none_raises():
    e = Engine.open(tempfile.mkdtemp(), {"kind": "none"})
    with pytest.raises(ProviderUnavailable):
        e.agent("anything")
    e.close()
