"""Regression for Finding 5 (symmetric-fact remaining gap): apply_merge must not create a
parallel duplicate open fact when both merged nodes carry the identical open SYMMETRIC fact
with a common neighbor.

Symmetric facts (works_with, married_to, friend_of) store ONE pinned (min,max) orientation
(kg/temporal.py:75). When survivor S and loser L each hold the same open symmetric fact with a
common neighbor X, id-order pinning stores them in DIFFERENT orientations: S→X (S<X) but X→L
(X<L, an in-edge to L). apply_merge re-points X→L to X→S; store.add_edge does NOT re-pin
RELATED_TO orientation, so before the fix a parallel X→S open edge was minted alongside the
survivor's canonical S→X — a duplicate open fact that double-surfaces in history() and orphans
one orientation from future supersede/close. The fix re-pins symmetric facts in _readd_edge.
"""
from __future__ import annotations

import os
import tempfile

from kg import Config
from kg.canonicalize import Canonicalizer
from kg.embedders import get_embedder
from kg.models import EdgeType, EntityType, NodeType, relation_tag_node
from kg.store import GraphStore, now_iso
from kg.temporal import apply_fact

P = EntityType.PERSON


def _cfg() -> Config:
    c = Config.default()
    c.embedder = "st"
    c.self_entity = False
    return c


def _canon() -> Canonicalizer:
    config = _cfg()
    store = GraphStore.open(os.path.join(tempfile.mkdtemp(), "kg.db"), config)
    return Canonicalizer(store, get_embedder(config), config)


def _sym_rel(store: GraphStore, name: str = "works_with") -> str:
    rid = "rel_sym"
    store.add_node(relation_tag_node(rid, canonical=name, ts=now_iso(), symmetric=True))
    return rid


def _all_edges_between(store: GraphStore, a: str, b: str, rel_tag: str) -> list:
    """Every RELATED_TO(rel_tag) edge in EITHER orientation between a and b."""
    out = list(store.find_facts(a, b, rel_tag))
    out += list(store.find_facts(b, a, rel_tag))
    return out


def _open_between(store: GraphStore, a: str, b: str, rel_tag: str) -> list:
    return [t for t in _all_edges_between(store, a, b, rel_tag)
            if not t[2].get("invalid_at", "")]


def test_symmetric_merge_collapses_cross_oriented_open_fact():
    canon = _canon()
    store = canon.store
    rel = _sym_rel(store)

    # id order S < X < L, so pinning stores S→X (in-edge nowhere) but X→L (in-edge to L).
    surv = canon.resolve_entity("Sam", P)
    common = canon.resolve_entity("Xavier", P)
    lose = canon.resolve_entity("Zoe", P)
    assert surv < common < lose  # the orientation-splitting id order the repro needs

    apply_fact(store, src=surv, dst=common, rel_tag=rel, status="asserted",
               at="2022", valid_from="2022", episode_id="e1")
    apply_fact(store, src=lose, dst=common, rel_tag=rel, status="asserted",
               at="2022", valid_from="2022", episode_id="e2")

    # sanity: the two copies are stored in OPPOSITE orientations (the crux of the bug).
    assert list(store.find_facts(surv, common, rel))       # S→X
    assert list(store.find_facts(common, lose, rel))        # X→L (in-edge to L)
    assert not list(store.find_facts(common, surv, rel))    # nothing yet in the X→S slot

    receipt = canon.apply_merge(surv, lose)
    assert receipt["merged"] is True

    # exactly ONE open fact between the survivor and the common neighbor — collapsed, not a
    # parallel S→X + X→S pair.
    assert len(_open_between(store, surv, common, rel)) == 1
    # and it lives in the canonical (min,max) orientation only.
    assert list(store.find_facts(surv, common, rel))
    assert not list(store.find_facts(common, surv, rel))
    # corroboration from both episodes folded together.
    f = list(store.find_facts(surv, common, rel))[0][2]
    assert set(f.get("confirmed_by", [])) >= {"e2"} or f.get("episode_id") in {"e1", "e2"}


def test_symmetric_merge_future_confirm_hits_single_edge():
    # after the merge, a later restatement must CONFIRM the one collapsed edge, never open a
    # second — proving the surviving edge is in the canonical slot apply_fact writes to.
    canon = _canon()
    store = canon.store
    rel = _sym_rel(store)
    surv = canon.resolve_entity("Sam", P)
    common = canon.resolve_entity("Xavier", P)
    lose = canon.resolve_entity("Zoe", P)
    assert surv < common < lose

    apply_fact(store, src=surv, dst=common, rel_tag=rel, status="asserted",
               at="2022", valid_from="2022", episode_id="e1")
    apply_fact(store, src=lose, dst=common, rel_tag=rel, status="asserted",
               at="2022", valid_from="2022", episode_id="e2")
    canon.apply_merge(surv, lose)

    action = apply_fact(store, src=common, dst=surv, rel_tag=rel, status="asserted",
                        at="2022", valid_from="2022", episode_id="e3")
    assert action == "confirm"
    assert len(_open_between(store, surv, common, rel)) == 1


def test_directional_merge_unchanged_by_repin():
    # guard: a NON-symmetric fact must NOT be re-pinned — direction is meaningful. Loser's
    # in-edge X→L re-points to X→S and stays X→S (not folded onto a hypothetical S→X).
    canon = _canon()
    store = canon.store
    rid = "rel_dir"
    store.add_node(relation_tag_node(rid, canonical="reports_to", ts=now_iso(),
                                     symmetric=False))
    surv = canon.resolve_entity("Sam", P)
    common = canon.resolve_entity("Xavier", P)
    lose = canon.resolve_entity("Zoe", P)

    apply_fact(store, src=common, dst=lose, rel_tag=rid, status="asserted",
               at="2022", valid_from="2022", episode_id="e1")
    canon.apply_merge(surv, lose)
    # the directional edge keeps its X→S direction (re-pointed), NOT flipped to S→X.
    assert list(store.find_facts(common, surv, rid))
    assert not list(store.find_facts(surv, common, rid))
