"""Event-representation tests (docs/PIPELINE.md sharp edge #1; docs/OFFLINE_EVAL.md
Round 2): event predicates write CLOSED [d,d] occurrence edges instead of open
[d, ∞) states, render as occurrences ("on d" / "d1 -> d2", never "since/until/ended"),
dedup same-day re-mentions against CLOSED edges, and coexist with old-format open
event edges (no migration). All behind config.event_facts (default OFF = byte-identical
pre-fix behavior). Fully offline/deterministic — mirrors tests/test_temporal.py.
"""
from __future__ import annotations

import os
import tempfile

import pytest

import kg.graph as kg_graph
from kg import Config
from kg.canonicalize import Canonicalizer, predicate_is_event
from kg.embedders import get_embedder
from kg.extractors import ScriptedExtractor
from kg.facts import FactIndex, FactLine
from kg.models import (EdgeType, EntityType, entity_node, relation_tag_node)
from kg.store import GraphStore, fact_active
from kg.temporal import apply_fact


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    """Same guard as tests/test_temporal.py: no key, no live extractor, ever."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(kg_graph, "get_extractor",
                        lambda config: ScriptedExtractor({}))


def cfg(event_facts: bool = True) -> Config:
    c = Config.default()
    c.embedder = "st"
    c.event_facts = event_facts
    return c


def _mini(event_facts: bool = True):
    store = GraphStore(cfg(event_facts))
    canon = Canonicalizer(store, get_embedder(cfg()), cfg(event_facts))
    for nid, name in [("e_me", "me"), ("e_park", "the park"),
                      ("e_japan", "Japan"), ("e_yoga", "yoga class")]:
        store.add_node(entity_node(nid, name=name, etype=EntityType.CONCEPT, ts="t"))
    return store, canon


def _one_fact(store, src, dst, rel):
    facts = list(store.find_facts(src, dst, rel))
    assert len(facts) == 1
    return facts[0][2]


# --------------------------------------------------------------------------- #
# lexicon stamping (canonicalization)
# --------------------------------------------------------------------------- #
def test_event_lexicon_stamping():
    store, canon = _mini()
    went = canon.resolve_relation("went_to")
    lives = canon.resolve_relation("lives_in")
    assert store.get_node(went).event is True
    assert store.get_node(lives).event is False
    # tense variants share the content key, so both stamp event
    assert predicate_is_event("visited") and predicate_is_event("visits")
    assert predicate_is_event("attended") and predicate_is_event("traveled_to")
    # "played"/"plays" is pointedly EXCLUDED: the stem collides with the habitual
    # state reading ("plays tennis on Tuesdays"), and a state misclassified as an
    # event is worse than a missed event.
    assert not predicate_is_event("played") and not predicate_is_event("plays")
    assert not predicate_is_event("employed_by") and not predicate_is_event("goes_to")


# --------------------------------------------------------------------------- #
# [d,d] write semantics
# --------------------------------------------------------------------------- #
def test_point_event_writes_closed_dd():
    store, canon = _mini()
    went = canon.resolve_relation("went_to")
    assert apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
                      status="asserted", at="2025-01-05") == "open"
    data = _one_fact(store, "e_me", "e_park", went)
    assert data["valid_at"] == "2025-01-05"
    assert data["invalid_at"] == "2025-01-05"          # [d, d] — over the day it happened
    assert data["event"] is True
    # closed by design: absent from the current view (served by the history/delta block)
    assert not fact_active(data, None)


def test_bounded_interval_passes_through_with_event_flag():
    store, canon = _mini()
    trav = canon.resolve_relation("traveled_to")
    apply_fact(store, src="e_me", dst="e_japan", rel_tag=trav, status="asserted",
               at="2023-10-20", valid_from="2023-11-01", valid_to="2023-11-14")
    data = _one_fact(store, "e_me", "e_japan", trav)
    assert data["valid_at"] == "2023-11-01" and data["invalid_at"] == "2023-11-14"
    assert data["event"] is True
    # the bounded window is a real as-of interval, exactly as before
    assert fact_active(data, "2023-11-05") and not fact_active(data, "2023-12-01")


def test_both_bounds_stated_is_event_shaped_by_construction():
    # a non-lexicon predicate arriving with BOTH bounds is an occurrence too
    store, canon = _mini()
    rel = canon.resolve_relation("rented_apartment_in")
    assert store.get_node(rel).event is False
    apply_fact(store, src="e_me", dst="e_japan", rel_tag=rel, status="asserted",
               at="2024-02-01", valid_from="2024-02-01", valid_to="2024-02-28")
    assert _one_fact(store, "e_me", "e_japan", rel)["event"] is True


# --------------------------------------------------------------------------- #
# confirm-on-closed dedup
# --------------------------------------------------------------------------- #
def test_same_day_double_mention_confirms_single_edge():
    store, canon = _mini()
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05", episode_id="ep_a")
    # same-day re-mention: the closed [d,d] edge must dedup, not duplicate
    assert apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
                      status="asserted", at="2025-01-05",
                      episode_id="ep_b") == "confirm"
    data = _one_fact(store, "e_me", "e_park", went)
    assert data["confirmed_by"] == ["ep_b"]


def test_different_date_opens_new_occurrence():
    store, canon = _mini()
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05")
    assert apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
                      status="asserted", at="2025-02-02") == "open"
    facts = list(store.find_facts("e_me", "e_park", went))
    assert len(facts) == 2
    assert all(d["valid_at"] == d["invalid_at"] and d["event"] for _v, _k, d in facts)
    assert {d["valid_at"] for _v, _k, d in facts} == {"2025-01-05", "2025-02-02"}


# --------------------------------------------------------------------------- #
# occurrence rendering
# --------------------------------------------------------------------------- #
def test_occurrence_rendering_never_uses_state_grammar():
    point = FactLine(src="me", rel="went_to", dst="the park",
                     valid_at="2025-01-05", invalid_at="2025-01-05", event=True)
    assert "(on 2025-01-05)" in point.render()
    bounded = FactLine(src="me", rel="traveled_to", dst="Japan",
                       valid_at="2023-11-01", invalid_at="2023-11-14", event=True)
    assert "(2023-11-01 -> 2023-11-14)" in bounded.render()
    for line in (point, bounded):
        r = line.render()
        assert "since" not in r and "until" not in r and "ended" not in r
        assert line.to_row()["status"] == "occurred"
    # same-day repeats keep the mentions counter visible
    repeated = FactLine(src="me", rel="went_to", dst="the park",
                        valid_at="2025-01-05", invalid_at="2025-01-05",
                        event=True, mentions=3)
    assert "on 2025-01-05" in repeated.render() and "3x" in repeated.render()


def test_engine_fact_row_reports_occurred(tmp_path):
    from kg import KnowledgeGraph
    g = KnowledgeGraph.open(os.path.join(str(tmp_path), "kg.db"), cfg())
    store, canon = g.store, g.canon
    store.add_node(entity_node("e_me", name="me", etype=EntityType.PERSON, ts="t"))
    store.add_node(entity_node("e_park", name="the park", etype=EntityType.CONCEPT, ts="t"))
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05")
    _v, _gkey, data = next(store.find_facts("e_me", "e_park", went))
    line = FactLine.from_edge(store, "e_me", "e_park", data)
    assert line.event and line.to_row()["status"] == "occurred"
    assert "(on 2025-01-05)" in line.to_row()["rendered"]


# --------------------------------------------------------------------------- #
# knob off + old-format coexistence
# --------------------------------------------------------------------------- #
def test_knob_off_keeps_open_state_write():
    store, canon = _mini(event_facts=False)
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05")
    assert apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
                      status="asserted", at="2025-02-02",
                      episode_id="ep_b") == "confirm"     # pre-fix collapse
    data = _one_fact(store, "e_me", "e_park", went)
    assert data["invalid_at"] == "" and not data.get("event")
    assert fact_active(data, None)                         # still (wrongly) current — as today
    line = FactLine.from_edge(store, "e_me", "e_park", data)
    assert "mentioned" in line.render() and line.to_row()["status"] == "asserted"


def test_old_format_events_coexist_and_round_trip(tmp_path):
    path = os.path.join(str(tmp_path), "kg.db")
    # 1) legacy store written with the knob OFF: open [d, ∞) event edge
    store = GraphStore.open(path, cfg(event_facts=False))
    canon = Canonicalizer(store, get_embedder(cfg()), cfg(event_facts=False))
    store.add_node(entity_node("e_me", name="me", etype=EntityType.PERSON, ts="t"))
    store.add_node(entity_node("e_park", name="the park", etype=EntityType.CONCEPT, ts="t"))
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05")
    store.save()
    # 2) reopen with the knob ON: legacy edge loads unchanged, still renders as today
    store2 = GraphStore.open(path, cfg(event_facts=True))
    _v, _k, legacy = next(store2.find_facts("e_me", "e_park", went))
    assert legacy["invalid_at"] == "" and not legacy.get("event")
    assert "mentioned 2025-01-05" in FactLine.from_edge(
        store2, "e_me", "e_park", legacy).render()
    # 3) a new occurrence on a different day coexists as a closed [d,d] edge
    assert apply_fact(store2, src="e_me", dst="e_park", rel_tag=went,
                      status="asserted", at="2025-03-09") == "open"
    facts = {d["valid_at"]: d for _v, _k, d in store2.find_facts("e_me", "e_park", went)}
    assert len(facts) == 2
    assert facts["2025-03-09"]["event"] and facts["2025-03-09"]["invalid_at"] == "2025-03-09"
    # 4) a same-day re-mention of the LEGACY open edge still dedups onto it
    assert apply_fact(store2, src="e_me", dst="e_park", rel_tag=went,
                      status="asserted", at="2025-01-05",
                      episode_id="ep_x") == "confirm"
    assert facts["2025-01-05"]["confirmed_by"] == ["ep_x"]
    # 5) the event flag round-trips through SQLite
    store2.save()
    store3 = GraphStore.open(path, cfg(event_facts=True))
    facts3 = {d["valid_at"]: bool(d.get("event"))
              for _v, _k, d in store3.find_facts("e_me", "e_park", went)}
    assert facts3 == {"2025-01-05": False, "2025-03-09": True}


# --------------------------------------------------------------------------- #
# history(): recency-ranked cap + delta block
# --------------------------------------------------------------------------- #
def test_history_cap_keeps_most_recent():
    store, canon = _mini()
    went = canon.resolve_relation("went_to")
    for i in range(1, 10):
        nid = f"e_p{i}"
        store.add_node(entity_node(nid, name=f"place {i}", etype=EntityType.CONCEPT, ts="t"))
        apply_fact(store, src="e_me", dst=nid, rel_tag=went,
                   status="asserted", at=f"2025-01-{i:02d}")
    hist = FactIndex(store).history(["e_me"], limit=3)
    # the cap keeps the MOST RECENT rows (pre-fix it kept the oldest), in time order
    assert [h.valid_at for h in hist] == ["2025-01-07", "2025-01-08", "2025-01-09"]


def test_history_cap_never_drops_closures_for_open_filler():
    # an OLD closure must keep its seat against newer open filler — closures are the
    # lines only the HISTORY block carries; open rows mostly restate FACTS
    store, canon = _mini(event_facts=False)
    emp = canon.resolve_relation("employed_by")
    store.add_node(entity_node("e_acme", name="Acme", etype=EntityType.ORG, ts="t"))
    apply_fact(store, src="e_me", dst="e_acme", rel_tag=emp,
               status="asserted", at="2019-02-01")
    apply_fact(store, src="e_me", dst="e_acme", rel_tag=emp, status="ended", at="2022-08-15")
    likes = canon.resolve_relation("subscribes_to")
    for i in range(1, 9):                                  # 8 recent OPEN facts
        nid = f"e_s{i}"
        store.add_node(entity_node(nid, name=f"feed {i}", etype=EntityType.CONCEPT, ts="t"))
        apply_fact(store, src="e_me", dst=nid, rel_tag=likes,
                   status="asserted", at=f"2025-01-{i:02d}")
    hist = FactIndex(store).history(["e_me"], limit=4)
    closed = [h for h in hist if h.invalid_at]
    assert len(hist) == 4
    assert [h.dst for h in closed] == ["Acme"]             # the 2019 closure survives


def test_history_all_lanes_serves_closed_delta_outside_state():
    from kg.rag import ContextBuilder
    from kg.retrieval import RetrievalResult
    store, canon = _mini()
    went = canon.resolve_relation("went_to")
    lives = canon.resolve_relation("lives_in")
    store.add_node(entity_node("e_den", name="Denver", etype=EntityType.PLACE, ts="t"))
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05")                       # closed event
    apply_fact(store, src="e_me", dst="e_den", rel_tag=lives,
               status="asserted", at="2023-05-01")                       # open state
    result = RetrievalResult(query="what have I done?", mode="ppr", objects=[])
    result.lane = "single"
    result.entity_ids = ["e_me"]
    c = cfg()

    _ids, _facts, blob = ContextBuilder(store, c).build(result)
    assert "HISTORY" not in blob                       # knob off: unchanged non-STATE lane

    c.history_all_lanes = True
    _ids, _facts, blob = ContextBuilder(store, c).build(result)
    assert "HISTORY" in blob
    assert "(on 2025-01-05)" in blob                   # the closed delta line
    hist_block = blob.split("HISTORY", 1)[1]
    assert "Denver" not in hist_block                  # open lines are FACTS-only (no dup)

    # STATE lane keeps its full closed+open block, exactly as before
    result.lane = "state"
    _ids, _facts, blob = ContextBuilder(store, c).build(result)
    hist_block = blob.split("HISTORY", 1)[1]
    assert "Denver" in hist_block and "on 2025-01-05" in hist_block


def test_history_all_lanes_delta_respects_as_of():
    from kg.rag import ContextBuilder
    from kg.retrieval import RetrievalResult
    store, canon = _mini()
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05")
    c = cfg()
    c.history_all_lanes = True
    result = RetrievalResult(query="what did I do?", mode="ppr", objects=[],
                             as_of="2024-12-01")       # BEFORE the event
    result.lane = "single"
    result.entity_ids = ["e_me"]
    _ids, _facts, blob = ContextBuilder(store, c).build(result)
    assert "2025-01-05" not in blob                    # not yet happened at T
    result.as_of = "2025-01-10"                        # AFTER the event
    _ids, _facts, blob = ContextBuilder(store, c).build(result)
    assert "(on 2025-01-05)" in blob                   # findable for a T question
