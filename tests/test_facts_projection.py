"""Round 6b — relational facts projection (config.facts_projection) + graph-tally
evidence lines (config.agg_evidence).

Both knobs default OFF and are QUERY-SIDE (not in INGEST_RELEVANT_FIELDS). The
projection writes two derived SQLite tables (facts_view / agg_view) rebuilt WHOLESALE on
every flush; the LOAD path never reads them. The tally lines are computed IN-MEMORY from
the believed RELATED_TO edges among the question's anchor entities and appended, capped
and caveated, to aggregate-shaped questions' context — EVIDENCE, never an oracle.

Fully offline/deterministic (no key, no live extractor) — mirrors tests/test_event_facts.py.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

import kg.graph as kg_graph
from kg import Config
from kg.canonicalize import Canonicalizer
from kg.embedders import get_embedder
from kg.extractors import ScriptedExtractor
from kg.ingest_cache import INGEST_RELEVANT_FIELDS
from kg.models import EntityType, entity_node
from kg.rag import ContextBuilder
from kg.retrieval import RetrievalResult
from kg.store import GraphStore
from kg.temporal import apply_fact


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(kg_graph, "get_extractor", lambda config: ScriptedExtractor({}))


def cfg(**over) -> Config:
    c = Config.default()
    c.embedder = "st"
    c.event_facts = True
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _store(path: str, **over) -> tuple[GraphStore, Canonicalizer]:
    c = cfg(**over)
    store = GraphStore(c, path=path)
    store._init_db()
    canon = Canonicalizer(store, get_embedder(cfg()), c)
    for nid, name in [("e_me", "me"), ("e_park", "the park"), ("e_gym", "the gym"),
                      ("e_japan", "Japan"), ("e_pizza", "pizza")]:
        store.add_node(entity_node(nid, name=name, etype=EntityType.CONCEPT, ts="t"))
    return store, canon


def tmp_db() -> str:
    return os.path.join(tempfile.mkdtemp(), "kg.db")


def _rows(path: str, table: str) -> list[tuple]:
    con = sqlite3.connect(path)
    try:
        return con.execute(f"SELECT * FROM {table}").fetchall()
    finally:
        con.close()


def _tables(path: str) -> set[str]:
    con = sqlite3.connect(path)
    try:
        return {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# knob is query-side (must NOT invalidate the ingest cache)
# --------------------------------------------------------------------------- #
def test_knobs_are_not_ingest_relevant():
    assert "facts_projection" not in INGEST_RELEVANT_FIELDS
    assert "agg_evidence" not in INGEST_RELEVANT_FIELDS


# --------------------------------------------------------------------------- #
# projection: written only when the knob is on
# --------------------------------------------------------------------------- #
def test_projection_absent_when_off():
    path = tmp_db()
    store, canon = _store(path, facts_projection=False)
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05")
    store.save()
    assert "facts_view" not in _tables(path)
    assert "agg_view" not in _tables(path)


def test_projection_created_on_flush_when_on():
    path = tmp_db()
    store, canon = _store(path, facts_projection=True)
    went = canon.resolve_relation("went_to")
    for d in ("2025-01-05", "2025-03-10", "2025-05-18"):
        apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
                   status="asserted", at=d)
    store.flush()
    assert {"facts_view", "agg_view"} <= _tables(path)
    # facts_view: one row per RELATED_TO edge (3 distinct dated occurrences)
    fv = _rows(path, "facts_view")
    assert len(fv) == 3
    assert all(r[0] == "me" and r[1] == "went_to" and r[2] == "the park" for r in fv)
    # agg_view: the pair grouped, n_occurrences = 3, spanning first..last date
    av = _rows(path, "agg_view")
    assert len(av) == 1
    src, rel, dst, n, first, last = av[0]
    assert (src, rel, dst) == ("me", "went_to", "the park")
    assert n == 3 and first == "2025-01-05" and last == "2025-05-18"


def test_mentions_counts_confirmations():
    path = tmp_db()
    store, canon = _store(path, facts_projection=True)
    likes = canon.resolve_relation("likes")   # non-event, undated: confirm-collapses
    apply_fact(store, src="e_me", dst="e_pizza", rel_tag=likes,
               status="asserted", at="2021", episode_id="ep1")
    apply_fact(store, src="e_me", dst="e_pizza", rel_tag=likes,
               status="asserted", at="2021", episode_id="ep2")
    store.flush()
    fv = [r for r in _rows(path, "facts_view") if r[2] == "pizza"]
    assert len(fv) == 1 and fv[0][-1] == 2          # mentions = 1 + len(confirmed_by)
    av = [r for r in _rows(path, "agg_view") if r[2] == "pizza"]
    assert av[0][3] == 2                             # n_occurrences = summed mentions


# --------------------------------------------------------------------------- #
# WHOLESALE rebuild: stale rows from a prior flush disappear
# --------------------------------------------------------------------------- #
def test_projection_rebuilt_wholesale_stale_rows_gone():
    path = tmp_db()
    store, canon = _store(path, facts_projection=True)
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05")
    store.flush()
    assert len(_rows(path, "facts_view")) == 1

    # a second, different occurrence to a DIFFERENT dst, then reflush — the rebuild must
    # reflect exactly the current edge set (2 rows), never accumulate.
    apply_fact(store, src="e_me", dst="e_gym", rel_tag=went,
               status="asserted", at="2025-02-02")
    store.flush()
    fv = _rows(path, "facts_view")
    assert len(fv) == 2
    assert {r[2] for r in fv} == {"the park", "the gym"}
    # agg_view likewise rebuilt wholesale: two distinct pairs, no ghost of the first-only flush
    assert {r[2] for r in _rows(path, "agg_view")} == {"the park", "the gym"}


# --------------------------------------------------------------------------- #
# retracted edges: present in facts_view (belief column), excluded from agg_view
# --------------------------------------------------------------------------- #
def test_retracted_excluded_from_agg_view():
    path = tmp_db()
    store, canon = _store(path, facts_projection=True)
    likes = canon.resolve_relation("likes")
    apply_fact(store, src="e_me", dst="e_pizza", rel_tag=likes,
               status="asserted", at="2021", episode_id="ep1")
    apply_fact(store, src="e_me", dst="e_pizza", rel_tag=likes,
               status="retracted", at="2022", episode_id="ep2")   # never true
    store.flush()
    fv = _rows(path, "facts_view")
    assert any(r[7] == "retracted" for r in fv)     # belief column still records it
    # the retracted pair contributes nothing to the believed-only aggregate
    assert all(r[2] != "pizza" for r in _rows(path, "agg_view"))


# --------------------------------------------------------------------------- #
# LOAD-path independence: delete the tables, reload → identical graph
# --------------------------------------------------------------------------- #
def _graph_snapshot(store: GraphStore):
    nodes = {nid: (n.ntype.value, n.name) for nid, n in store.nodes.items()}
    edges = sorted(
        (u, v, d.get("etype"), d.get("rel_tag") or "", d.get("valid_at", ""),
         d.get("invalid_at", ""), d.get("belief", "asserted"), int(d.get("seq", 0)))
        for u, v, d in store.all_edges())
    return nodes, edges


def test_load_path_ignores_projection_tables():
    path = tmp_db()
    store, canon = _store(path, facts_projection=True)
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05")
    apply_fact(store, src="e_me", dst="e_gym", rel_tag=went,
               status="asserted", at="2025-02-02")
    store.flush()
    assert {"facts_view", "agg_view"} <= _tables(path)

    with_tables = _graph_snapshot(GraphStore.open(path, cfg(facts_projection=True)))

    con = sqlite3.connect(path)
    con.execute("DROP TABLE facts_view")
    con.execute("DROP TABLE agg_view")
    con.commit()
    con.close()
    assert "facts_view" not in _tables(path)

    without_tables = _graph_snapshot(GraphStore.open(path, cfg(facts_projection=True)))
    assert with_tables == without_tables


# --------------------------------------------------------------------------- #
# restored-cache store: no tables on disk, first flush creates them cleanly
# --------------------------------------------------------------------------- #
def test_restored_cache_first_flush_creates_tables():
    path = tmp_db()
    # phase 1: written with the knob OFF (the cache was built by an earlier run) — no tables
    store, canon = _store(path, facts_projection=False)
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05")
    store.save()
    assert "facts_view" not in _tables(path)

    # phase 2: reopened with the knob ON (as after try_restore). A fresh mutation + flush
    # must create the tables cleanly over a db that never had them.
    store2 = GraphStore.open(path, cfg(facts_projection=True))
    canon2 = Canonicalizer(store2, get_embedder(cfg()), store2.config)
    went2 = canon2.resolve_relation("went_to")
    apply_fact(store2, src="e_me", dst="e_gym", rel_tag=went2,
               status="asserted", at="2025-02-02")
    store2.flush()
    assert {"facts_view", "agg_view"} <= _tables(path)
    assert {r[2] for r in _rows(path, "facts_view")} == {"the park", "the gym"}


def test_projection_only_reflush_with_nothing_dirty():
    """The knob-on, nothing-dirty path still (re)builds the tables — it is not treated as a
    no-op the way an off-knob idle save is."""
    path = tmp_db()
    store, canon = _store(path, facts_projection=False)
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went,
               status="asserted", at="2025-01-05")
    store.save()
    assert "facts_view" not in _tables(path)

    store2 = GraphStore.open(path, cfg(facts_projection=True))
    store2.save()   # nothing dirty, but the projection must materialize
    assert {"facts_view", "agg_view"} <= _tables(path)
    assert len(_rows(path, "facts_view")) == 1


# --------------------------------------------------------------------------- #
# tally evidence lines (config.agg_evidence)
# --------------------------------------------------------------------------- #
def _build(store, agg_evidence, query, entity_ids):
    c = cfg(agg_evidence=agg_evidence)
    result = RetrievalResult(query=query, mode="ppr", objects=[])
    result.entity_ids = entity_ids   # type: ignore[attr-defined]
    return ContextBuilder(store, c).build(result)[2]


def _park_gym_store():
    path = tmp_db()
    store, canon = _store(path, facts_projection=False)
    went = canon.resolve_relation("went_to")
    for d in ("2025-01-05", "2025-02-02", "2025-03-10", "2025-04-14", "2025-05-18"):
        apply_fact(store, src="e_me", dst="e_park", rel_tag=went, status="asserted", at=d)
    for d in ("2025-01-20", "2025-06-01"):
        apply_fact(store, src="e_me", dst="e_gym", rel_tag=went, status="asserted", at=d)
    return store


def test_tally_present_and_ordered_when_on():
    store = _park_gym_store()
    blob = _build(store, True, "How many times did I go to the park?", ["e_me"])
    assert "GRAPH TALLIES (may be incomplete; verify against the episodes):" in blob
    assert "me --went_to--> the park: 5 occurrences (2025-01-05 -> 2025-05-18)" in blob
    assert "me --went_to--> the gym: 2 occurrences (2025-01-20 -> 2025-06-01)" in blob
    # ordered by occurrence count desc: park (5) before gym (2)
    assert blob.index("the park: 5") < blob.index("the gym: 2")


def test_tally_absent_when_off_and_append_only():
    store = _park_gym_store()
    q = "How many times did I go to the park?"
    off = _build(store, False, q, ["e_me"])
    on = _build(store, True, q, ["e_me"])
    assert "GRAPH TALLIES" not in off
    # both knobs off is byte-identical to on-minus-appendix (feature only APPENDS)
    assert on.startswith(off)
    assert on[len(off):].lstrip().startswith("GRAPH TALLIES")


def test_tally_absent_for_non_aggregate_question():
    store = _park_gym_store()
    blob = _build(store, True, "Where did I go on the weekend?", ["e_me"])
    assert "GRAPH TALLIES" not in blob


def test_tally_capped_at_ten_lines():
    path = tmp_db()
    store, canon = _store(path, facts_projection=False)
    went = canon.resolve_relation("went_to")
    # 15 distinct destinations, descending occurrence counts, so ordering + cap are visible
    for i in range(15):
        did = f"e_dst{i}"
        store.add_node(entity_node(did, name=f"place{i:02d}", etype=EntityType.CONCEPT, ts="t"))
        for j in range(15 - i):           # place00 has 15 occurrences, place14 has 1
            apply_fact(store, src="e_me", dst=did, rel_tag=went, status="asserted",
                       at=f"2025-{1 + j:02d}-01" if j < 12 else f"2026-{j - 11:02d}-01")
    blob = _build(store, True, "How many places did I go in all?", ["e_me"])
    tally_lines = [ln for ln in blob.splitlines() if ln.startswith("me --went_to-->")]
    assert len(tally_lines) == 10                       # capped
    # the top-10 by count are place00..place09 (15..6 occurrences); place14 (1) is cut
    assert "place00" in blob and "place09" in blob
    assert "place14" not in blob
    assert "me --went_to--> place00: 15 occurrences" in tally_lines[0]


def test_tally_excludes_retracted():
    path = tmp_db()
    store, canon = _store(path, facts_projection=False)
    likes = canon.resolve_relation("likes")
    apply_fact(store, src="e_me", dst="e_pizza", rel_tag=likes, status="asserted",
               at="2021", episode_id="ep1")
    apply_fact(store, src="e_me", dst="e_pizza", rel_tag=likes, status="retracted",
               at="2022", episode_id="ep2")
    blob = _build(store, True, "How much pizza do I have in all?", ["e_me"])
    # the retracted pair must never appear as a tally
    assert "pizza" not in blob or "GRAPH TALLIES" not in blob
