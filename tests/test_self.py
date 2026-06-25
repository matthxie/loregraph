"""Personal-web "self" anchor tests (optional first-person resolution).

Fully offline/deterministic: a hashing embedder + scripted/heuristic extractor, no API
key, no network. Verifies the feature's contract — first-person resolution ON, the
byte-for-byte-unchanged OFF-path, offline fact formation on the self anchor, and a
save/load round-trip that re-routes "me" after reopening. Run: python -m pytest -q
"""
from __future__ import annotations

import os
import tempfile

from kg import Config, KnowledgeGraph
from kg.canonicalize import Canonicalizer
from kg.embedders import get_embedder
from kg.extractors import ScriptedExtractor
from kg.models import SELF_ENTITY_ID, EdgeType, EntityType, NodeType
from kg.store import GraphStore
from kg.synthetic import personal_stream


def _cfg(self_entity: bool = False, self_name: str = "self") -> Config:
    c = Config.default()
    c.embedder = "hashing"
    c.extractor = "heuristic"
    c.self_entity = self_entity
    c.self_name = self_name
    return c


def _tmp() -> str:
    return os.path.join(tempfile.mkdtemp(), "kg.db")


def _canon(config: Config) -> Canonicalizer:
    store = GraphStore.open(_tmp(), config)
    return Canonicalizer(store, get_embedder(config), config)


# --------------------------------------------------------------------------- #
# resolution ON
# --------------------------------------------------------------------------- #
def test_first_person_resolves_to_self_anchor():
    canon = _canon(_cfg(self_entity=True, self_name="self"))
    P = EntityType.PERSON
    sid = canon.resolve_entity("me", P)
    assert sid == SELF_ENTITY_ID
    # every first-person surface form (any casing) collapses onto the one anchor
    for surface in ("I", "me", "Myself", "my", "Mine"):
        assert canon.resolve_entity(surface, P) == SELF_ENTITY_ID
    node = canon.store.get_node(SELF_ENTITY_ID)
    assert node is not None
    assert node.ntype == NodeType.ENTITY
    assert node.entity_type == EntityType.PERSON
    assert node.name == "self"
    # the self anchor carries NO entity embedding — it is found by pronoun routing, never by
    # similarity (which is also what stops a name-colliding real entity from merging into it)
    assert canon.store.vectors.get("entity", SELF_ENTITY_ID) is None


def test_self_name_is_configurable():
    canon = _canon(_cfg(self_entity=True, self_name="Jude"))
    assert canon.resolve_entity("me", EntityType.PERSON) == SELF_ENTITY_ID
    assert canon.store.get_node(SELF_ENTITY_ID).name == "Jude"


def test_self_name_does_not_capture_a_real_entity():
    """Only first-person pronouns route to self; the display name is NOT a resolution key,
    so a --self that collides with a real third party can't silently merge into self."""
    canon = _canon(_cfg(self_entity=True, self_name="Becky"))
    P = EntityType.PERSON
    assert canon.resolve_entity("me", P) == SELF_ENTITY_ID
    assert canon.resolve_entity("Becky", P) != SELF_ENTITY_ID   # the real Becky stays herself


def test_changed_self_name_refreshes_display_on_reopen():
    path = _tmp()
    g = KnowledgeGraph.open(path, _cfg(self_entity=True, self_name="self"))
    g.canon.resolve_entity("me", EntityType.PERSON)
    g.save()
    # reopen with a different --self → _reindex/_ensure_self refreshes the persisted name
    g2 = KnowledgeGraph.open(path, _cfg(self_entity=True, self_name="Jude"))
    assert g2.store.get_node(SELF_ENTITY_ID).name == "Jude"
    assert g2.canon.resolve_entity("me", EntityType.PERSON) == SELF_ENTITY_ID


# --------------------------------------------------------------------------- #
# OFF-path (default) — proves the feature is inert when off
# --------------------------------------------------------------------------- #
def test_off_path_me_is_normal_entity():
    canon = _canon(_cfg(self_entity=False))   # default
    P = EntityType.PERSON
    me = canon.resolve_entity("me", P)
    assert me is not None
    assert me != SELF_ENTITY_ID                # NOT the special anchor
    assert not canon.store.has_node(SELF_ENTITY_ID)
    # "me" and "you" are independent entities (no special self-anchoring)
    you = canon.resolve_entity("you", P)
    assert me != you


def test_off_path_default_config_self_false():
    # the dataclass default must keep the feature off
    assert Config.default().self_entity is False
    assert Config.default().self_name == "self"


# --------------------------------------------------------------------------- #
# offline ingest — the self anchor behaves like any other entity in the pipeline
# --------------------------------------------------------------------------- #
def _becky_id(g) -> str:
    return next(n.id for n in g.store.nodes_of_type(NodeType.ENTITY) if n.name == "Becky")


def test_ingest_personal_stream_forms_self_facts():
    g = KnowledgeGraph.open(_tmp(), _cfg(self_entity=True))
    items, table = personal_stream()
    g.extractor = ScriptedExtractor(table)
    g.ingest(items)

    assert g.store.has_node(SELF_ENTITY_ID)
    becky = _becky_id(g)

    # a RELATED_TO fact self --had_coffee_with--> Becky formed (directed out-edge)
    out_facts = list(g.store.find_facts(SELF_ENTITY_ID))
    targets = {dst for dst, _k, _d in out_facts}
    assert becky in targets
    out_labels = {g.store.get_node(d["rel_tag"]).name
                  for _v, _k, d in out_facts if d.get("rel_tag")}
    assert "had_coffee_with" in out_labels
    # the later episodes also attach facts to the anchor — works_with is symmetric so the
    # graph pins it to one orientation; check the self anchor's facts in BOTH directions.
    all_labels = set(out_labels)
    for nbr, d in g.store.neighbors(SELF_ENTITY_ID, etypes={EdgeType.RELATED_TO},
                                    direction="both"):
        if d.get("rel_tag"):
            all_labels.add(g.store.get_node(d["rel_tag"]).name)
    assert "works_with" in all_labels

    # the self anchor accrues mentions across episodes (RESOLVES_TO in-edges)
    in_mentions = [nbr for nbr, _d in g.store.neighbors(
        SELF_ENTITY_ID, etypes={EdgeType.RESOLVES_TO}, direction="in")]
    assert len(in_mentions) >= 2
    eps = {g.store.episode_of(m) for m in in_mentions}
    assert len([e for e in eps if e]) >= 2     # stable identity across multiple items


# --------------------------------------------------------------------------- #
# save / load round-trip
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip_routes_first_person():
    path = _tmp()
    g = KnowledgeGraph.open(path, _cfg(self_entity=True))
    items, table = personal_stream()
    g.extractor = ScriptedExtractor(table)
    g.ingest(items)
    g.save()

    # reopen the persisted store with the feature on — _reindex must re-route "me"
    g2 = KnowledgeGraph.open(path, _cfg(self_entity=True))
    assert g2.store.has_node(SELF_ENTITY_ID)
    assert g2.canon.resolve_entity("me", EntityType.PERSON) == SELF_ENTITY_ID
    assert g2.canon.resolve_entity("I", EntityType.PERSON) == SELF_ENTITY_ID
    # the persisted facts survived
    facts = list(g2.store.find_facts(SELF_ENTITY_ID))
    assert facts
