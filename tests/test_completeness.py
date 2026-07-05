"""Tests for kg/completeness.py — the extraction-completeness metrics
(spikes/completeness/REPORT.md, automated). Synthetic fixtures only; no live LLM calls
(tier 2's LLM call is exercised via a fake client, mirroring tests/test_dashboard.py).

Run: python -m pytest tests/test_completeness.py -q
"""
from __future__ import annotations

import json
import types

from kg.completeness import (classify_occurrences, enumerate_occurrences_llm,
                             find_amounts_in_text, is_aggregate_question,
                             node_amount_set, normalize_amount,
                             quantity_capture_for_question, question_shape,
                             summarize_tier1, summarize_tier2)
from kg.config import Config
from kg.models import Edge, EdgeType, EntityType, Provenance, entity_node
from kg.store import GraphStore


def cfg():
    c = Config.default()
    c.embed_dim = 4
    return c


# --------------------------------------------------------------------------- #
# question shape
# --------------------------------------------------------------------------- #
def test_is_aggregate_question():
    assert is_aggregate_question("How many doctor's appointments did I go to in March?")
    assert is_aggregate_question("What is the total amount I spent on luxury items?")
    assert is_aggregate_question("How much did I earn altogether?")
    assert not is_aggregate_question("What is my dog's name?")


def test_question_shape():
    assert question_shape("How many fun runs did I miss?") == "count"
    assert question_shape("What is the total amount I earned?") == "sum"
    assert question_shape("How much did I spend in all?") == "sum"
    assert question_shape("What's my favorite color?") == "other"


# --------------------------------------------------------------------------- #
# tier 1 — regex / normalization
# --------------------------------------------------------------------------- #
def test_normalize_amount():
    assert normalize_amount("$1,300.00") == "1300"
    assert normalize_amount("1300") == "1300"
    assert normalize_amount("$495") == "495"
    assert normalize_amount("2.50") == "2.5"


def test_find_amounts_in_text_dedup_and_dollars_word():
    text = ("I sold a scarf for $50 at the market. Later that day I made another $50 on "
           "a necklace. I also earned 200 dollars from a custom order.")
    assert find_amounts_in_text(text) == ["50", "200"]


def test_find_amounts_in_text_empty_when_no_quantities():
    assert find_amounts_in_text("I went to the market and bought a scarf.") == []


def test_quantity_capture_for_question_matches_report_shape():
    """Synthetic version of the REPORT's SUM-question finding: 3 amounts mentioned in
    evidence text, only 1 shows up as a node -> capture_rate 1/3."""
    store = GraphStore(cfg())
    store.add_node(entity_node("n1", name="$500", etype=EntityType.CONCEPT, ts="t"))
    evidence_text = "I earned $500 at the spring market, $300 at the summer market, and $150 at the fall market."
    rec = quantity_capture_for_question("q1", "What is the total amount I earned at the markets?",
                                        evidence_text, store)
    assert rec is not None
    assert rec["shape"] == "sum"
    assert rec["amounts_in_text"] == 3
    assert rec["amounts_in_graph"] == 1
    assert rec["capture_rate"] == round(1 / 3, 3)


def test_quantity_capture_none_when_not_aggregate_question():
    store = GraphStore(cfg())
    assert quantity_capture_for_question("q1", "What is my dog's name?", "$500", store) is None


def test_quantity_capture_none_when_no_amounts_in_evidence():
    store = GraphStore(cfg())
    assert quantity_capture_for_question(
        "q1", "How many times did I go running?", "I went running with my friend.", store
    ) is None


def test_node_amount_set():
    store = GraphStore(cfg())
    store.add_node(entity_node("n1", name="$800", etype=EntityType.CONCEPT, ts="t"))
    store.add_node(entity_node("n2", name="Alan Turing", etype=EntityType.PERSON, ts="t"))
    assert node_amount_set(store) == {"800"}


def test_summarize_tier1_pools_across_questions():
    recs = [
        {"question_id": "a", "shape": "sum", "amounts_in_text": 3, "amounts_in_graph": 1,
         "capture_rate": 0.333},
        {"question_id": "b", "shape": "sum", "amounts_in_text": 3, "amounts_in_graph": 2,
         "capture_rate": 0.667},
    ]
    s = summarize_tier1(recs)
    assert s["n_questions"] == 2
    assert s["amounts_in_text"] == 6 and s["amounts_in_graph"] == 3
    assert s["capture_rate"] == 0.5


def test_summarize_tier1_none_when_no_records():
    assert summarize_tier1([]) is None


# --------------------------------------------------------------------------- #
# tier 2 — occurrence classification (CAPTURED / COLLAPSED / MISSING)
# --------------------------------------------------------------------------- #
def _add_fact(store, src, dst, episode_id, rel_tag="rel"):
    store.add_edge(Edge(src, dst, EdgeType.RELATED_TO, Provenance.EXTRACTED, 1.0, 1.0,
                        rel_tag=rel_tag, episode_id=episode_id))


def test_classify_captured_two_distinct_occurrences():
    """Mirrors REPORT's 00ca467f (doctor's appointments): two occurrences, each its own
    edge/episode -> both CAPTURED."""
    store = GraphStore(cfg())
    for nid, name in [("smith", "Dr. Smith"), ("thompson", "Dr. Thompson"), ("user", "User")]:
        store.add_node(entity_node(nid, name=name, etype=EntityType.PERSON, ts="t"))
    _add_fact(store, "user", "smith", "ep_q1__s1", rel_tag="diagnose")
    _add_fact(store, "user", "thompson", "ep_q1__s2", rel_tag="follow")

    occs = [{"session_id": "s1", "quote": "I saw Dr. Smith for bronchitis", "amount": None},
           {"session_id": "s2", "quote": "Follow-up with Dr. Thompson on March 20th", "amount": None}]
    out = classify_occurrences(store, "q1", occs)
    assert [o["status"] for o in out] == ["CAPTURED", "CAPTURED"]


def test_classify_collapsed_compound_node():
    """Mirrors REPORT's 2788b940 (Zumba): two weekly occurrences flattened into ONE
    compound node 'Tuesdays and Thursdays' -> both COLLAPSED."""
    store = GraphStore(cfg())
    store.add_node(entity_node("zumba", name="fitness class", etype=EntityType.CONCEPT, ts="t"))
    store.add_node(entity_node("days", name="Tuesdays and Thursdays", etype=EntityType.CONCEPT, ts="t"))
    _add_fact(store, "zumba", "days", "ep_q2__s1", rel_tag="occurs_on")

    occs = [{"session_id": "s1", "quote": "I do Zumba every Tuesday", "amount": None},
           {"session_id": "s1", "quote": "I also do Zumba on Thursdays", "amount": None}]
    out = classify_occurrences(store, "q2", occs)
    assert [o["status"] for o in out] == ["COLLAPSED", "COLLAPSED"]
    assert out[0]["node"] == out[1]["node"] == "Tuesdays and Thursdays"


def test_classify_missing_amount_never_extracted():
    """Mirrors REPORT's SUM questions: a dollar amount mentioned nowhere in the graph
    -> MISSING."""
    store = GraphStore(cfg())
    store.add_node(entity_node("market", name="Farmers Market", etype=EntityType.PLACE, ts="t"))
    _add_fact(store, "market", "market", "ep_q3__s1", rel_tag="held_at")  # unrelated fact

    occs = [{"session_id": "s1", "quote": "I earned $495 selling scarves", "amount": 495}]
    out = classify_occurrences(store, "q3", occs)
    assert out[0]["status"] == "MISSING"


def test_classify_captured_amount_matches_node():
    store = GraphStore(cfg())
    store.add_node(entity_node("amt", name="$495", etype=EntityType.CONCEPT, ts="t"))
    store.add_node(entity_node("market", name="Farmers Market", etype=EntityType.PLACE, ts="t"))
    _add_fact(store, "market", "amt", "ep_q3__s1", rel_tag="earned")

    occs = [{"session_id": "s1", "quote": "I earned $495 selling scarves", "amount": 495}]
    out = classify_occurrences(store, "q3", occs)
    assert out[0]["status"] == "CAPTURED" and out[0]["node"] == "$495"


def test_classify_ignores_structural_edges():
    """MENTIONED_IN/RESOLVES_TO etc. carry episode_id too, but aren't facts — the
    classifier must not treat them as evidence of capture."""
    store = GraphStore(cfg())
    store.add_node(entity_node("m", name="Dr. Smith", etype=EntityType.PERSON, ts="t"))
    store.add_node(entity_node("ep", name="episode", etype=EntityType.CONCEPT, ts="t"))
    store.add_edge(Edge("m", "ep", EdgeType.MENTIONED_IN, Provenance.EXTRACTED, 1.0, 1.0,
                        episode_id="ep_q1__s1"))
    occs = [{"session_id": "s1", "quote": "I saw Dr. Smith", "amount": None}]
    out = classify_occurrences(store, "q1", occs)
    assert out[0]["status"] == "MISSING"


def test_summarize_tier2_aggregate_and_by_shape():
    classified_by_question = {
        "q1": ("count", [{"status": "CAPTURED"}, {"status": "CAPTURED"}]),
        "q2": ("sum", [{"status": "MISSING"}, {"status": "MISSING"}, {"status": "CAPTURED"}]),
    }
    s = summarize_tier2(classified_by_question)
    assert s["n"] == 5 and s["captured"] == 3 and s["missing"] == 2 and s["collapsed"] == 0
    assert s["by_shape"]["count"]["pct_captured"] == 1.0
    assert s["by_shape"]["sum"]["pct_missing"] == round(2 / 3, 3)


def test_summarize_tier2_none_when_empty():
    assert summarize_tier2({}) is None


# --------------------------------------------------------------------------- #
# tier 2 — LLM enumeration call (fake client, no network)
# --------------------------------------------------------------------------- #
class _FakeCompletionsClient:
    def __init__(self, content):
        self._content = content
        self.chat = self
        self.completions = self
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        message = types.SimpleNamespace(content=self._content)
        choice = types.SimpleNamespace(message=message)
        usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        return types.SimpleNamespace(choices=[choice], usage=usage)


def test_enumerate_occurrences_llm_parses_and_meters():
    from kg.metering import UsageMeter
    payload = json.dumps({"occurrences": [{"session_id": "s1", "quote": "I earned $495",
                                           "amount": 495}], "notes": ""})
    client = _FakeCompletionsClient(payload)
    meter = UsageMeter()
    occs = enumerate_occurrences_llm(client, "gpt-4o-mini", "q1", "How much did I earn?",
                                     [{"session_id": "s1", "date": "2023", "text": "I earned $495"}],
                                     meter=meter)
    assert occs == [{"session_id": "s1", "quote": "I earned $495", "amount": 495}]
    assert meter.totals()["llm_calls"] == 1
    assert client.calls[0]["model"] == "gpt-4o-mini"


def test_enumerate_occurrences_llm_returns_empty_on_bad_json():
    client = _FakeCompletionsClient("not json")
    occs = enumerate_occurrences_llm(client, "gpt-4o-mini", "q1", "How much?",
                                     [{"session_id": "s1", "date": "", "text": "x"}])
    assert occs == []
