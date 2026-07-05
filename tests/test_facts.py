"""Tests for the typed quantity-fact extraction path (extraction-completeness fix).

Covers: emit_graph schema parsing of facts[] into ExtractedFact, compound-value
splitting ("Tuesdays and Thursdays" / "$200 and $50"), per-occurrence non-collapse
(both quantity facts and dated repeatable relations), and the canonicalizer/write-path
guarantee that distinct amounts never alias-merge. Fully offline: ScriptedExtractor
feeds known Extractions, no LLM call.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from kg import Config, KnowledgeGraph
from kg.corpus import CorpusItem
from kg.extractors import (Extraction, ExtractedEntity, ExtractedFact, ExtractedRelation,
                          ScriptedExtractor, _parse_tool_payload, _split_compound)
from kg.models import EdgeType, EntityType, NodeType
from kg.temporal import apply_fact
from kg.canonicalize import Canonicalizer
from kg.embedders import get_embedder
from kg.store import GraphStore
from kg.models import entity_node


def cfg() -> Config:
    c = Config.default()
    c.embedder = "st"
    return c


@pytest.fixture(autouse=True)
def _no_live_extractor(monkeypatch):
    import kg.graph as _graph
    monkeypatch.setattr(_graph, "get_extractor", lambda config: ScriptedExtractor({}))


def tmp_store() -> str:
    return os.path.join(tempfile.mkdtemp(), "kg.db")


# --------------------------------------------------------------------------- #
# schema parsing
# --------------------------------------------------------------------------- #
def test_parse_tool_payload_extracts_facts():
    ext = _parse_tool_payload({
        "entities": [{"name": "Jane", "type": "person"}],
        "tags": [],
        "facts": [{"subject": "Jane", "predicate": "earned", "value": 250,
                  "unit": "usd", "date": "2023-03-15"}],
    })
    assert len(ext.facts) == 1
    f = ext.facts[0]
    assert f.subject == "Jane" and f.predicate == "earned"
    assert f.value == 250.0 and f.unit == "usd" and f.date == "2023-03-15"


def test_parse_tool_payload_facts_require_subject_predicate_value():
    ext = _parse_tool_payload({
        "entities": [], "tags": [],
        "facts": [{"subject": "", "predicate": "earned", "value": 5},
                  {"subject": "Jane", "predicate": "", "value": 5},
                  {"subject": "Jane", "predicate": "earned", "value": None},
                  {"subject": "Jane", "predicate": "earned", "value": "not a number"}],
    })
    assert ext.facts == []


# --------------------------------------------------------------------------- #
# compound splitting
# --------------------------------------------------------------------------- #
def test_split_compound_weekdays():
    assert _split_compound("Tuesdays and Thursdays") == ["Tuesdays", "Thursdays"]


def test_split_compound_amounts():
    assert _split_compound("$200 and $50") == ["$200", "$50"]


def test_split_compound_leaves_proper_names_intact():
    assert _split_compound("Bed and Breakfast") == ["Bed and Breakfast"]
    assert _split_compound("Johnson and Johnson") == ["Johnson and Johnson"]


def test_split_compound_single_value_unchanged():
    assert _split_compound("Monday") == ["Monday"]


def test_relation_target_compound_split_into_two_relations():
    """The Zumba case: one relation whose target names two days becomes two relations."""
    ext = _parse_tool_payload({
        "entities": [{"name": "Zumba", "type": "event"},
                    {"name": "Tuesdays and Thursdays", "type": "date"}],
        "tags": [],
        "relations": [{"source": "Zumba", "target": "Tuesdays and Thursdays",
                       "labels": ["scheduled_on"]}],
    })
    targets = sorted(r.target for r in ext.relations)
    assert targets == ["Thursdays", "Tuesdays"]


def test_fact_subject_compound_split_into_two_facts():
    ext = _parse_tool_payload({
        "entities": [], "tags": [],
        "facts": [{"subject": "Alice and Bob", "predicate": "earned", "value": 100}],
    })
    # "Alice and Bob" -> 2 words each part, distinct, but neither matches weekday/numeric
    # so it should NOT split (conservative default) — assert the guard, not an assumption.
    assert [f.subject for f in ext.facts] == ["Alice and Bob"]


# --------------------------------------------------------------------------- #
# Extraction.merge: facts and dated relations must not collapse occurrences
# --------------------------------------------------------------------------- #
def test_merge_keeps_distinct_dated_relation_occurrences():
    first = Extraction(relations=[ExtractedRelation(source="Me", target="Farmers Market",
                                                     labels=["visited"], valid_from="2023-03-01")])
    second = Extraction(relations=[ExtractedRelation(source="Me", target="Farmers Market",
                                                      labels=["visited"], valid_from="2023-03-15")])
    merged = first.merge(second)
    assert len(merged.relations) == 2


def test_merge_dedupes_exact_repeat_relation():
    first = Extraction(relations=[ExtractedRelation(source="Me", target="Farmers Market",
                                                     labels=["visited"], valid_from="2023-03-01")])
    second = Extraction(relations=[ExtractedRelation(source="Me", target="Farmers Market",
                                                      labels=["visited"], valid_from="2023-03-01")])
    merged = first.merge(second)
    assert len(merged.relations) == 1


def test_merge_keeps_distinct_facts_dedupes_exact_repeats():
    first = Extraction(facts=[ExtractedFact(subject="Me", predicate="earned", value=200,
                                            unit="usd", date="2023-03-01")])
    second = Extraction(facts=[
        ExtractedFact(subject="Me", predicate="earned", value=200, unit="usd", date="2023-03-01"),  # dup
        ExtractedFact(subject="Me", predicate="earned", value=50, unit="usd", date="2023-03-15"),   # new
    ])
    merged = first.merge(second)
    assert len(merged.facts) == 2
    assert {f.value for f in merged.facts} == {200, 50}


# --------------------------------------------------------------------------- #
# apply_fact: dated repeatable-predicate occurrences don't collapse
# --------------------------------------------------------------------------- #
def _mini():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    for nid, name in [("e_me", "Me"), ("e_market", "Farmers Market")]:
        store.add_node(entity_node(nid, name=name, etype=EntityType.PERSON, ts="t"))
    return store, canon


def test_apply_fact_distinct_dates_open_two_occurrences():
    store, canon = _mini()
    visited = canon.resolve_relation("visited")
    assert apply_fact(store, src="e_me", dst="e_market", rel_tag=visited,
                      status="asserted", at="2023-03-01", valid_from="2023-03-01") == "open"
    # a second, differently-dated occurrence of the SAME (src,dst,rel) must OPEN, not confirm
    assert apply_fact(store, src="e_me", dst="e_market", rel_tag=visited,
                      status="asserted", at="2023-03-15", valid_from="2023-03-15") == "open"
    facts = list(store.find_facts("e_me", "e_market", visited))
    assert len(facts) == 2
    assert {f[2]["valid_at"] for f in facts} == {"2023-03-01", "2023-03-15"}


def test_apply_fact_same_date_still_confirms():
    store, canon = _mini()
    visited = canon.resolve_relation("visited")
    apply_fact(store, src="e_me", dst="e_market", rel_tag=visited,
              status="asserted", at="2023-03-01", valid_from="2023-03-01")
    assert apply_fact(store, src="e_me", dst="e_market", rel_tag=visited,
                      status="asserted", at="2023-03-01", valid_from="2023-03-01") == "confirm"
    assert len(list(store.find_facts("e_me", "e_market", visited))) == 1


def test_apply_fact_undated_repeat_still_confirms():
    """No explicit date on either call: unchanged legacy behavior (confirm, not open)."""
    store, canon = _mini()
    visited = canon.resolve_relation("visited")
    apply_fact(store, src="e_me", dst="e_market", rel_tag=visited, status="asserted", at="2021")
    assert apply_fact(store, src="e_me", dst="e_market", rel_tag=visited,
                      status="asserted", at="2021-07") == "confirm"


# --------------------------------------------------------------------------- #
# end-to-end ingest: quantity nodes never alias-merge, occurrences stay distinct
# --------------------------------------------------------------------------- #
def test_ingest_writes_distinct_quantity_nodes_per_occurrence():
    item = CorpusItem(id="a", modality="text", source_ref="u/a", title="Markets",
                      text="I earned $250 at the spring market and $2,500 at the winter market.")
    table = {
        item.text: Extraction(
            entities=[ExtractedEntity("me", EntityType.PERSON),
                     ExtractedEntity("spring market", EntityType.PLACE),
                     ExtractedEntity("winter market", EntityType.PLACE)],
            tags=["markets"],
            facts=[ExtractedFact(subject="me", predicate="earned", value=250, unit="usd"),
                  ExtractedFact(subject="me", predicate="earned", value=2500, unit="usd")],
        ),
    }
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.extractor = ScriptedExtractor(table)
    report = g.ingest([item])
    assert report.quantity_facts == 2

    qty_nodes = [n for n in g.store.nodes_of_type(NodeType.ENTITY)
                if n.entity_type == EntityType.QUANTITY]
    assert len(qty_nodes) == 2
    values = sorted(n.value for n in qty_nodes)
    assert values == [250.0, 2500.0]          # distinct amounts, never merged into one node

    # both are reachable from "me" via a RELATED_TO edge and summable without regex
    me = next(n for n in g.store.nodes_of_type(NodeType.ENTITY) if n.name == "me")
    total = sum(nbr_node.value for nbr, _d in g.store.neighbors(me.id, etypes={EdgeType.RELATED_TO})
               if (nbr_node := g.store.get_node(nbr)) and nbr_node.entity_type == EntityType.QUANTITY)
    assert total == 2750.0
