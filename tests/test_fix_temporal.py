"""Regression tests for Finding 6 (kg/temporal.py): out-of-order ingest of a repeatable
predicate must not collapse distinct occurrences.

engine-v0 had narrowed the "new dated occurrence" guard to LATER-only, so an EARLIER-dated
distinct occurrence ingested after a later one fell into the confirm branch and had its
valid_at rewritten by the widen-to-earliest code — two real visits collapsed into one and the
later occurrence's date vanished. The fix restores distinct-date semantics for REPEATABLE
predicates (any date different from every open occurrence opens a new one, earlier or later)
while keeping widen-to-earliest confirm for non-repeatable functional/symmetric predicates.

Fully offline: canonicalizer stamps cardinality from the predicate surface, no LLM call.
"""
from __future__ import annotations

from kg import Config
from kg.canonicalize import Canonicalizer
from kg.embedders import get_embedder
from kg.models import EntityType, entity_node
from kg.store import GraphStore
from kg.temporal import apply_fact


def cfg() -> Config:
    c = Config.default()
    c.embedder = "st"
    return c


def _mini():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    for nid, name in [("e_me", "Me"), ("e_paris", "Paris")]:
        store.add_node(entity_node(nid, name=name, etype=EntityType.PERSON, ts="t"))
    return store, canon


def test_repeatable_earlier_dated_occurrence_opens_new_fact():
    """The core finding: a later occurrence ingested first, then an EARLIER-dated distinct
    occurrence — both must survive as separate open facts, neither date overwritten."""
    store, canon = _mini()
    visited = canon.resolve_relation("visited")   # neither functional nor symmetric → repeatable
    assert apply_fact(store, src="e_me", dst="e_paris", rel_tag=visited,
                      status="asserted", at="2024-05-10", valid_from="2024-05-10") == "open"
    # earlier-dated visit arrives out of order — must OPEN, not confirm-and-widen
    assert apply_fact(store, src="e_me", dst="e_paris", rel_tag=visited,
                      status="asserted", at="2024-03-02", valid_from="2024-03-02") == "open"
    facts = list(store.find_facts("e_me", "e_paris", visited))
    assert len(facts) == 2
    assert {f[2]["valid_at"] for f in facts} == {"2024-03-02", "2024-05-10"}


def test_repeatable_later_dated_occurrence_still_opens():
    """Later-dated distinct occurrence keeps opening a new fact (unchanged direction)."""
    store, canon = _mini()
    visited = canon.resolve_relation("visited")
    assert apply_fact(store, src="e_me", dst="e_paris", rel_tag=visited,
                      status="asserted", at="2024-03-02", valid_from="2024-03-02") == "open"
    assert apply_fact(store, src="e_me", dst="e_paris", rel_tag=visited,
                      status="asserted", at="2024-05-10", valid_from="2024-05-10") == "open"
    assert len(list(store.find_facts("e_me", "e_paris", visited))) == 2


def test_functional_earlier_date_widens_to_earliest_confirm():
    """Non-repeatable path: a single-valued state re-asserted with an EARLIER date is the SAME
    fact begun earlier — confirm and widen valid_at to the earliest start, not a new occurrence."""
    store, canon = _mini()
    lives_in = canon.resolve_relation("lives_in")   # functional → NOT repeatable
    assert apply_fact(store, src="e_me", dst="e_paris", rel_tag=lives_in,
                      status="asserted", at="2024-05-10", valid_from="2024-05-10") == "open"
    assert apply_fact(store, src="e_me", dst="e_paris", rel_tag=lives_in,
                      status="asserted", at="2024-03-02", valid_from="2024-03-02") == "confirm"
    facts = list(store.find_facts("e_me", "e_paris", lives_in))
    assert len(facts) == 1
    assert facts[0][2]["valid_at"] == "2024-03-02"   # widened to earliest known start


def test_functional_later_date_confirms_not_new_occurrence():
    """Non-repeatable path: a later differing date is a restatement of the ongoing state —
    confirm (single fact), never a second occurrence."""
    store, canon = _mini()
    lives_in = canon.resolve_relation("lives_in")
    assert apply_fact(store, src="e_me", dst="e_paris", rel_tag=lives_in,
                      status="asserted", at="2024-03-02", valid_from="2024-03-02") == "open"
    assert apply_fact(store, src="e_me", dst="e_paris", rel_tag=lives_in,
                      status="asserted", at="2024-05-10", valid_from="2024-05-10") == "confirm"
    facts = list(store.find_facts("e_me", "e_paris", lives_in))
    assert len(facts) == 1
    assert facts[0][2]["valid_at"] == "2024-03-02"   # earliest start retained
