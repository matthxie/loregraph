"""Regression tests for two merge-layer defects:

Finding 4 — resolve_entity's L2 fallback used to inspect ONLY hits[0]. When a
type-incompatible same-name node ('Jordan' PERSON) permanently shadowed a compatible
duplicate ('Jordan' PLACE), every later mention minted a fresh duplicate — unbounded.

Finding 5 — Canonicalizer.apply_merge re-added a loser's facts with no dedup, so when
both nodes carried the SAME open fact the survivor ended up with two parallel open
duplicates that double-surface in FactIndex and get independently confirmed/closed.
"""
from __future__ import annotations

import os
import tempfile

from kg import Config
from kg.canonicalize import Canonicalizer
from kg.embedders import get_embedder
from kg.models import Belief, Edge, EdgeType, EntityType, NodeType, Provenance
from kg.store import GraphStore

P = EntityType.PERSON
PLACE = EntityType.PLACE
ORG = EntityType.ORG


def _cfg() -> Config:
    c = Config.default()
    c.embedder = "st"          # deterministic local bge embedder
    c.self_entity = False
    return c


def _tmp() -> str:
    return os.path.join(tempfile.mkdtemp(), "kg.db")


def _canon() -> Canonicalizer:
    config = _cfg()
    store = GraphStore.open(_tmp(), config)
    return Canonicalizer(store, get_embedder(config), config)


def _entities_named(store: GraphStore, name: str) -> list[str]:
    return [n.id for n in store.nodes_of_type(NodeType.ENTITY)
            if n.name.lower() == name.lower()]


# --------------------------------------------------------------------------- #
# Finding 4 — no unbounded minting on a type-collision name
# --------------------------------------------------------------------------- #
def test_type_collision_name_does_not_mint_unbounded_duplicates():
    canon = _canon()
    # 'Jordan' the person is minted first; 'Jordan' the place L1-collides on the shared
    # proper-noun key, is type-incompatible, and is minted as a distinct anchor.
    person = canon.resolve_entity("Jordan", P)
    place = canon.resolve_entity("Jordan", PLACE)
    assert person != place
    assert canon.store.get_node(person).entity_type == P
    assert canon.store.get_node(place).entity_type == PLACE

    # Every FURTHER 'Jordan' PLACE mention must resolve back onto the existing PLACE anchor,
    # NOT fall through to hits[0] (the PERSON, which wins the id tie-break) and mint anew.
    for _ in range(5):
        assert canon.resolve_entity("Jordan", PLACE) == place

    # exactly two 'Jordan' entity nodes exist — the person and the place, nothing more.
    assert len(_entities_named(canon.store, "Jordan")) == 2


def test_type_collision_place_first_then_person():
    # symmetry: the shadowing node being the PLACE (minted first) must not block the PERSON.
    canon = _canon()
    place = canon.resolve_entity("Jordan", PLACE)
    person = canon.resolve_entity("Jordan", P)
    assert place != person
    for _ in range(3):
        assert canon.resolve_entity("Jordan", P) == person
        assert canon.resolve_entity("Jordan", PLACE) == place
    assert len(_entities_named(canon.store, "Jordan")) == 2


def test_same_name_same_type_still_reuses_one_anchor():
    # guard against over-minting: identical name + compatible type is still ONE node (L1).
    canon = _canon()
    a = canon.resolve_entity("Morgan", P)
    b = canon.resolve_entity("Morgan", P)
    c = canon.resolve_entity("Morgan", EntityType.OTHER)  # OTHER refines onto the person
    assert a == b == c
    assert len(_entities_named(canon.store, "Morgan")) == 1


# --------------------------------------------------------------------------- #
# Finding 5 — merging two nodes that carry the identical open fact collapses it
# --------------------------------------------------------------------------- #
def _add_fact(store: GraphStore, src: str, dst: str, *, valid_at: str, invalid_at: str = "",
              confidence: float = 1.0, episode_id: str = "", confirmed_by=None,
              rel_tag: str = "works_at") -> None:
    store.add_edge(Edge(
        src=src, dst=dst, etype=EdgeType.RELATED_TO, rel_tag=rel_tag,
        provenance=Provenance.EXTRACTED, confidence=confidence, weight=1.0,
        valid_at=valid_at, invalid_at=invalid_at, belief=Belief.ASSERTED,
        episode_id=episode_id, confirmed_by=list(confirmed_by or [])))


def _open_facts(store: GraphStore, src: str, dst: str, rel_tag: str = "works_at") -> list[dict]:
    return [d for _v, _k, d in store.find_facts(src, dst, rel_tag, open_only=True)]


def test_merge_collapses_identical_open_fact():
    canon = _canon()
    store = canon.store
    surv = canon.resolve_entity("Robert", P)
    lose = canon.resolve_entity("Bob", P)
    acme = canon.resolve_entity("Acme", ORG)

    # both duplicates carry the SAME open fact (same rel_tag, endpoint, valid_at, open).
    _add_fact(store, surv, acme, valid_at="2026-01-01", confidence=0.6, episode_id="e1",
              confirmed_by=["e1"])
    _add_fact(store, lose, acme, valid_at="2026-01-01", confidence=0.9, episode_id="e2",
              confirmed_by=["e2"])

    receipt = canon.apply_merge(surv, lose)
    assert receipt["merged"] is True

    facts = _open_facts(store, surv, acme)
    assert len(facts) == 1                       # collapsed, not two parallel opens
    f = facts[0]
    # the loser's edge was stronger (0.9 > 0.6) → its confidence + provenance won,
    # and the corroboration lists were unioned.
    assert f["confidence"] == 0.9
    assert f["episode_id"] == "e2"
    assert set(f["confirmed_by"]) == {"e1", "e2"}


def test_merge_keeps_stronger_survivor_metadata():
    canon = _canon()
    store = canon.store
    surv = canon.resolve_entity("Kate", P)
    lose = canon.resolve_entity("Katherine", P)
    acme = canon.resolve_entity("Globex", ORG)

    _add_fact(store, surv, acme, valid_at="2026-01-01", confidence=0.95, episode_id="s1",
              confirmed_by=["s1"])
    _add_fact(store, lose, acme, valid_at="2026-01-01", confidence=0.4, episode_id="l1",
              confirmed_by=["l1"])

    canon.apply_merge(surv, lose)
    facts = _open_facts(store, surv, acme)
    assert len(facts) == 1
    f = facts[0]
    assert f["confidence"] == 0.95               # survivor's stronger claim retained
    assert f["episode_id"] == "s1"
    assert set(f["confirmed_by"]) == {"s1", "l1"}  # corroboration still unioned


def test_merge_does_not_collapse_distinct_facts():
    canon = _canon()
    store = canon.store
    surv = canon.resolve_entity("Alice", P)
    lose = canon.resolve_entity("Alicia", P)
    acme = canon.resolve_entity("Initech", ORG)

    # different valid_at start → genuinely different fact occurrences, must stay parallel.
    _add_fact(store, surv, acme, valid_at="2025-01-01")
    _add_fact(store, lose, acme, valid_at="2026-06-01")
    canon.apply_merge(surv, lose)
    assert len(_open_facts(store, surv, acme)) == 2


def test_merge_does_not_collapse_open_onto_closed():
    canon = _canon()
    store = canon.store
    surv = canon.resolve_entity("Dan", P)
    lose = canon.resolve_entity("Daniel", P)
    acme = canon.resolve_entity("Umbrella", ORG)

    # same valid_at, but one is already CLOSED (invalid_at set) — different status, keep both.
    _add_fact(store, surv, acme, valid_at="2026-01-01", invalid_at="2026-03-01")
    _add_fact(store, lose, acme, valid_at="2026-01-01")   # still open
    canon.apply_merge(surv, lose)
    # one open + one closed edge survive; find_facts (no open filter) yields both.
    all_facts = [d for _v, _k, d in store.find_facts(surv, acme, "works_at")]
    assert len(all_facts) == 2
    assert len(_open_facts(store, surv, acme)) == 1
