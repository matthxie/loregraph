"""Round 7a — fact-line embeddings (config.fact_vectors): statement-granularity
retrieval vectors stored under kind="fact".

Two surface families, both kind="fact", namespaced by node-id prefix (fact: / factagg:):
a STATEMENT surface per believed RELATED_TO edge ("<src> <rel> <dst>", deduped by text)
and a distilled AGGREGATE surface per recurring (src,rel,dst) group ("<src> <rel> <dst> N
times from <first> to <last>"). Keyed by SURFACE HASH so parallel occurrences dedupe to
one vector and a canonical rename re-embeds the new surface + orphans the old.

fact_vectors is an INGEST-cache field (hashed only when ON) — but a $0 local backfill
enriches existing off-caches in place without a paid re-ingest.

Fully offline/deterministic — no key, a FakeEmbedder for the storage/reconciliation tests
(embedding QUALITY is irrelevant to storage) and the real local bge only on the one
end-to-end ingest path. Mirrors tests/test_facts_projection.py.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import tempfile

import numpy as np
import pytest

import kg.graph as kg_graph
from kg import Config
from kg.canonicalize import Canonicalizer
from kg.embedders import get_embedder
from kg.extractors import ScriptedExtractor
from kg.fact_vectors import (FACT_KIND, backfill_fact_vectors, current_surfaces,
                             statement_surface, sync_fact_vectors)
from kg.graph import KnowledgeGraph
from kg.ingest_cache import INGEST_RELEVANT_FIELDS, _config_digest
from kg.models import EntityType, entity_node
from kg.store import GraphStore
from kg.temporal import apply_fact


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(kg_graph, "get_extractor", lambda config: ScriptedExtractor({}))


class FakeEmbedder:
    """Deterministic, network-free embedder for the storage tests. Distinct text -> a
    distinct unit vector derived from its hash; dim is small (vectors are stored per-kind,
    so this never has to match the store's episode/entity dim)."""
    name = "fake"
    dim = 16

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            v = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32)
            v = v / (np.linalg.norm(v) or 1.0)
            out.append(v)
        return np.asarray(out, dtype=np.float32)


def cfg(**over) -> Config:
    c = Config.default()
    c.embedder = "st"
    c.event_facts = True
    for k, v in over.items():
        setattr(c, k, v)
    return c


def tmp_db() -> str:
    return os.path.join(tempfile.mkdtemp(), "kg.db")


def _store(path: str, **over) -> tuple[GraphStore, Canonicalizer]:
    c = cfg(**over)
    store = GraphStore(c, path=path)
    store._init_db()
    canon = Canonicalizer(store, get_embedder(cfg()), c)
    for nid, name in [("e_me", "me"), ("e_park", "the park"), ("e_gym", "the gym"),
                      ("e_japan", "Japan"), ("e_pizza", "pizza")]:
        store.add_node(entity_node(nid, name=name, etype=EntityType.CONCEPT, ts="t"))
    return store, canon


def _vec_kinds(path: str) -> set[str]:
    con = sqlite3.connect(path)
    try:
        return {r[0] for r in con.execute("SELECT DISTINCT kind FROM vectors")}
    finally:
        con.close()


def _vec_ids(path: str, kind: str) -> set[str]:
    con = sqlite3.connect(path)
    try:
        return {r[0] for r in con.execute(
            "SELECT node_id FROM vectors WHERE kind=?", (kind,))}
    finally:
        con.close()


def _park_gym_store(**over):
    """5 park visits (one deduped surface + a 5-time aggregate) and 2 gym visits."""
    path = tmp_db()
    store, canon = _store(path, **over)
    went = canon.resolve_relation("went_to")
    for d in ("2025-01-05", "2025-02-02", "2025-03-10", "2025-04-14", "2025-05-18"):
        apply_fact(store, src="e_me", dst="e_park", rel_tag=went, status="asserted", at=d)
    for d in ("2025-01-20", "2025-06-01"):
        apply_fact(store, src="e_me", dst="e_gym", rel_tag=went, status="asserted", at=d)
    return path, store, canon


# --------------------------------------------------------------------------- #
# surfaces: deterministic, name-resolved, no dates/ids
# --------------------------------------------------------------------------- #
def test_statement_surface_name_resolved_and_dateless():
    path, store, canon = _park_gym_store()
    # one park edge, in stored orientation
    u, v, d = next(iter(store.g.edges(data=True)))
    surf = statement_surface(store, u, v, d)
    assert surf == "me went_to the park"          # names, not ids
    assert "2025" not in surf and "e_me" not in surf   # no dates, no node ids


def test_surfaces_deterministic_and_deduped():
    path, store, canon = _park_gym_store()
    stmt1, agg1 = current_surfaces(store)
    stmt2, agg2 = current_surfaces(store)
    assert stmt1 == stmt2 and agg1 == agg2        # deterministic
    # 5 parallel park occurrences dedupe to ONE statement surface; gym is a second.
    assert set(stmt1.values()) == {"me went_to the park", "me went_to the gym"}


def test_aggregate_only_for_multi_occurrence_pairs():
    path, store, canon = _park_gym_store()
    likes = canon.resolve_relation("likes")
    apply_fact(store, src="e_me", dst="e_pizza", rel_tag=likes,   # single occurrence
               status="asserted", at="2025-01-01")
    _stmt, agg = current_surfaces(store)
    aggs = set(agg.values())
    assert "me went_to the park 5 times from 2025-01-05 to 2025-05-18" in aggs
    assert "me went_to the gym 2 times from 2025-01-20 to 2025-06-01" in aggs
    # the single-occurrence pizza pair gets NO aggregate surface
    assert not any("pizza" in a for a in aggs)


def test_aggregate_counts_confirmations_not_just_edges():
    """A confirm-collapsed pair (one edge, N-1 confirmations) still aggregates as N times."""
    path = tmp_db()
    store, canon = _store(path)
    likes = canon.resolve_relation("likes")
    for ep in ("ep1", "ep2", "ep3"):
        apply_fact(store, src="e_me", dst="e_pizza", rel_tag=likes,
                   status="asserted", at="2021", episode_id=ep)   # same-date confirm-collapse
    _stmt, agg = current_surfaces(store)
    # one edge with two confirmations → n_occurrences = 3, valid_at widened to 2021
    assert set(agg.values()) == {"me likes pizza 3 times from 2021 to 2021"}


def test_aggregate_undated_has_no_span():
    """A recurring pair with no dated occurrence renders no 'from..to' span."""
    path = tmp_db()
    store, canon = _store(path)
    likes = canon.resolve_relation("likes")
    for ep in ("ep1", "ep2"):
        apply_fact(store, src="e_me", dst="e_pizza", rel_tag=likes,
                   status="asserted", at="", episode_id=ep)
    _stmt, agg = current_surfaces(store)
    assert set(agg.values()) == {"me likes pizza 2 times"}


# --------------------------------------------------------------------------- #
# sync: stores under kind="fact", statement + aggregate distinguished by id prefix
# --------------------------------------------------------------------------- #
def test_sync_stores_both_families_under_fact_kind():
    path, store, canon = _park_gym_store()
    rep = sync_fact_vectors(store, FakeEmbedder(), prune=True)
    ids = set(store.vectors.ids(FACT_KIND))
    stmt_ids = {i for i in ids if i.startswith("fact:")}
    agg_ids = {i for i in ids if i.startswith("factagg:")}
    assert rep["statements"] == len(stmt_ids) == 2      # park, gym
    assert rep["aggregates"] == len(agg_ids) == 2       # park 5x, gym 2x
    assert rep["added"] == 4 and rep["total"] == 4
    # persistence round-trips under the "fact" kind (relation-canon vectors coexist)
    store.save()
    assert FACT_KIND in _vec_kinds(path)
    assert _vec_ids(path, FACT_KIND) == ids


def test_sync_idempotent():
    path, store, canon = _park_gym_store()
    fe = FakeEmbedder()
    sync_fact_vectors(store, fe, prune=True)
    rep2 = sync_fact_vectors(store, fe, prune=True)      # nothing new
    assert rep2["added"] == 0 and rep2["removed"] == 0


# --------------------------------------------------------------------------- #
# knob OFF: no "fact" kind, no extra writes (byte-identical writes)
# --------------------------------------------------------------------------- #
def test_knob_off_writes_no_fact_kind():
    path, store, canon = _park_gym_store(fact_vectors=False)
    store.save()
    # apply_fact writes no vectors, and nothing else does either → no vectors table rows
    assert FACT_KIND not in _vec_kinds(path)
    assert store.vectors.ids(FACT_KIND) == []


# --------------------------------------------------------------------------- #
# end-to-end ingest: batch embed at ingest ONLY when the knob is on
# --------------------------------------------------------------------------- #
def _becky_graph(**over) -> KnowledgeGraph:
    from kg.synthetic import becky_stream
    g = KnowledgeGraph.open(tmp_db(), cfg(**over))
    items, table = becky_stream()
    g.extractor = ScriptedExtractor(table)
    report = g.ingest(items)
    g.save()
    return g, report


def test_ingest_embeds_facts_when_on():
    g, report = _becky_graph(fact_vectors=True)
    ids = g.store.vectors.ids(FACT_KIND)
    assert ids, "ingest with fact_vectors=on must embed statement surfaces"
    assert report.fact_vectors == len(ids)
    # persisted under kind="fact"
    assert FACT_KIND in _vec_kinds(g.store.path)
    # surfaces are name-resolved (becky_stream asserts lives_in/works_with/employed_by)
    stmt, _agg = current_surfaces(g.store)
    assert any("Becky" in s for s in stmt.values())


def test_ingest_no_facts_when_off():
    g, report = _becky_graph(fact_vectors=False)
    assert g.store.vectors.ids(FACT_KIND) == []
    assert report.fact_vectors == 0
    assert FACT_KIND not in _vec_kinds(g.store.path)


# --------------------------------------------------------------------------- #
# rename on canonical merge → re-embed the new surface, orphan the old
# --------------------------------------------------------------------------- #
def test_rename_on_merge_reembeds():
    path, store, canon = _park_gym_store()
    fe = FakeEmbedder()
    sync_fact_vectors(store, fe, prune=True)
    stmt, _agg = current_surfaces(store)
    old_park_id = next(i for i, s in stmt.items() if s == "me went_to the park")
    assert store.vectors.get(FACT_KIND, old_park_id) is not None

    # a canonicalization merge renames the endpoint: "the park" -> "Green Park"
    store.get_node("e_park").name = "Green Park"
    store.touch_node("e_park")

    rep = sync_fact_vectors(store, fe, prune=True)
    new_stmt, _agg = current_surfaces(store)
    new_ids = set(store.vectors.ids(FACT_KIND))

    # the renamed statement + its aggregate changed text → 2 new, 2 orphaned
    assert rep["added"] == 2 and rep["removed"] == 2
    assert old_park_id not in new_ids                       # stale surface pruned
    assert "me went_to Green Park" in set(new_stmt.values())
    assert all(store.vectors.get(FACT_KIND, i) is not None for i in new_stmt)

    # the prune persists as a DELETE, not a lingering row
    store.save()
    assert old_park_id not in _vec_ids(path, FACT_KIND)
    assert set(store.vectors.ids(FACT_KIND)) == _vec_ids(path, FACT_KIND)


# --------------------------------------------------------------------------- #
# backfill: additive, idempotent, incremental
# --------------------------------------------------------------------------- #
def test_backfill_idempotent_and_incremental():
    path, store, canon = _park_gym_store()
    fe = FakeEmbedder()
    r1 = backfill_fact_vectors(store, fe)
    assert r1["added"] == 4 and r1["removed"] == 0
    r2 = backfill_fact_vectors(store, fe)                   # idempotent
    assert r2["added"] == 0 and r2["removed"] == 0

    # incremental: a NEW fact adds only its own surfaces, nothing re-embedded
    went = canon.resolve_relation("went_to")
    apply_fact(store, src="e_me", dst="e_japan", rel_tag=went, status="asserted",
               at="2025-07-01")
    r3 = backfill_fact_vectors(store, fe)
    assert r3["added"] == 1 and r3["removed"] == 0          # one new single-occurrence stmt
    assert r3["statements"] == 3


def test_backfill_is_purely_additive_no_prune():
    """Backfill must NOT prune orphans (a rename leaves a stale vector rather than deleting)
    — additive-only is what keeps it safe to run against a cached store."""
    path, store, canon = _park_gym_store()
    fe = FakeEmbedder()
    backfill_fact_vectors(store, fe)
    before = set(store.vectors.ids(FACT_KIND))
    store.get_node("e_park").name = "Green Park"
    store.touch_node("e_park")
    rep = backfill_fact_vectors(store, fe)
    assert rep["removed"] == 0                              # nothing pruned
    assert before <= set(store.vectors.ids(FACT_KIND))      # old ids still present


# --------------------------------------------------------------------------- #
# backfill on a COPY of a real cached benchmark store (skipped if none present)
# --------------------------------------------------------------------------- #
def _first_cache_store() -> str | None:
    cache_dir = os.path.join("store", "cache")
    if not os.path.isdir(cache_dir):
        return None
    for name in sorted(os.listdir(cache_dir)):
        if name.endswith(".db"):
            return os.path.join(cache_dir, name)
    return None


def test_backfill_on_real_cache_copy():
    src = _first_cache_store()
    if src is None:
        pytest.skip("no store/cache/*.db present (real benchmark caches)")
    dst = os.path.join(tempfile.mkdtemp(), "cache_copy.db")
    shutil.copy(src, dst)
    # opened knob-OFF, as the cache was built — backfill is an explicit action, not gated
    g = KnowledgeGraph.open(dst, cfg(fact_vectors=False))
    r1 = g.backfill_fact_vectors()
    g.save()
    assert r1["added"] > 0 and r1["statements"] > 0        # a real store has believed facts
    assert FACT_KIND in _vec_kinds(dst)
    # idempotent on a reopen: the same surfaces are already present
    g2 = KnowledgeGraph.open(dst, cfg(fact_vectors=False))
    r2 = g2.backfill_fact_vectors()
    assert r2["added"] == 0 and r2["removed"] == 0


# --------------------------------------------------------------------------- #
# ingest-cache back-compat: knob-off digest unchanged; on changes it
# --------------------------------------------------------------------------- #
def test_fact_vectors_is_ingest_relevant_but_hash_only_when_on():
    assert "fact_vectors" in INGEST_RELEVANT_FIELDS
    base = Config.default()                       # fact_vectors off
    assert _config_digest(base) == _config_digest(Config.default())   # stable
    on = Config.default()
    on.fact_vectors = True
    assert _config_digest(on) != _config_digest(base)                 # ON re-keys the store


def test_load_roundtrip_preserves_fact_vectors():
    path, store, canon = _park_gym_store()
    sync_fact_vectors(store, FakeEmbedder(), prune=True)
    store.save()
    saved = set(store.vectors.ids(FACT_KIND))
    reloaded = GraphStore.open(path, cfg())
    assert set(reloaded.vectors.ids(FACT_KIND)) == saved
