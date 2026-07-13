"""Fork-parity regression tests (fork-parity-spec: brainbrain fork logic ported to
engine-v0). One test per item in the spec's Testing-requirements matrix:

  D1 proper-noun entity_key           D2 type-compat merge guard
  D3 alias persistence across reload  D4 works_at ≠ works_with
  E2 retracted end-to-end             E1 dispute gating
  E3 symmetric-functional supersede   C3 once_/past_ no longer terminate
  C1 "concepts" payload back-compat   C1 term/preference type aliases
  F2 category stamp + THING upgrade   F3 salted-hash dedup
  A1 provider auto-detect ordering

Fully offline: ScriptedExtractor feeds known Extractions, the provider probes are
monkeypatched, and the only model loaded is the local st embedder.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from kg import Config, KnowledgeGraph
from kg.canonicalize import (Canonicalizer, entity_key, normalize_key,
                             relation_merge_vetoed)
from kg.corpus import CorpusItem
from kg.embedders import get_embedder
from kg.extractors import (ExtractedEntity, ExtractedRelation, Extraction,
                           ScriptedExtractor, _parse_tool_payload)
from kg.facts import FactIndex
from kg.models import (Belief, EntityCategory, EntityType, NodeType, entity_node,
                       relation_tag_node)
from kg.store import GraphStore, fact_active
from kg.temporal import apply_fact


def cfg() -> Config:
    c = Config.default()
    c.embedder = "st"
    return c


def tmp_store() -> str:
    return os.path.join(tempfile.mkdtemp(), "kg.db")


@pytest.fixture(autouse=True)
def _no_live_extractor(monkeypatch):
    import kg.graph as _graph
    monkeypatch.setattr(_graph, "get_extractor", lambda config: ScriptedExtractor({}))


def fresh_canon(store: GraphStore | None = None) -> tuple[GraphStore, Canonicalizer]:
    store = store or GraphStore(cfg())
    return store, Canonicalizer(store, get_embedder(cfg()), cfg())


def temporal_store() -> GraphStore:
    """Entity/relation-tag scaffolding for driving apply_fact directly."""
    store = GraphStore(cfg())
    for nid, name in [("e_me", "Me"), ("e_ann", "Ann"), ("e_bob", "Bob"),
                      ("e_tor", "Toronto"), ("e_ber", "Berlin")]:
        store.add_node(entity_node(nid, name=name, etype=EntityType.PERSON, ts="t"))
    store.add_node(relation_tag_node("rel_spouse", canonical="spouse_of", ts="t",
                                     functional=True, symmetric=True))
    store.add_node(relation_tag_node("rel_lives", canonical="lives_in", ts="t",
                                     functional=True))
    store.add_node(relation_tag_node("rel_knows", canonical="knows", ts="t"))
    return store


# --------------------------------------------------------------------------- #
# D1 — proper-noun entity_key: no last-token depluralisation for names
# --------------------------------------------------------------------------- #
def test_entity_key_proper_noun_guard():
    # normalize_key alone would collide the cat "Socks" with the garment "Sock" …
    assert normalize_key("Socks") == normalize_key("Sock") == "sock"
    # … the typed key keeps capitalized person/place/org NAMES verbatim
    assert entity_key("Socks", EntityType.PERSON) == "socks"
    assert entity_key("Sock", EntityType.PERSON) == "sock"
    assert entity_key("Paris", EntityType.PLACE) == "paris"      # not "pari"
    # common-noun / concept surfaces keep singularization
    assert entity_key("socks", EntityType.CONCEPT) == entity_key("sock", EntityType.CONCEPT)
    # a lowercase surface is not a proper noun even under a proper type
    assert entity_key("socks", EntityType.PERSON) == "sock"


# --------------------------------------------------------------------------- #
# D2 — type-compatibility merge guard
# --------------------------------------------------------------------------- #
def test_type_compat_guard_jordan_person_vs_place():
    _store, canon = fresh_canon()
    person = canon.resolve_entity("Jordan", EntityType.PERSON)
    place = canon.resolve_entity("Jordan", EntityType.PLACE)
    assert person != place                       # identical surface, contradictory types
    # each type keeps resolving to its own anchor
    assert canon.resolve_entity("Jordan", EntityType.PERSON) == person


def test_type_compat_other_upgrades_to_specific():
    store, canon = fresh_canon()
    a = canon.resolve_entity("Acme", EntityType.OTHER)
    assert store.get_node(a).entity_type == EntityType.OTHER
    assert canon.resolve_entity("Acme", EntityType.ORG) == a     # OTHER is compatible
    assert store.get_node(a).entity_type == EntityType.ORG       # …and got upgraded
    # a later untyped mention does not downgrade it back
    assert canon.resolve_entity("Acme", EntityType.OTHER) == a
    assert store.get_node(a).entity_type == EntityType.ORG


# --------------------------------------------------------------------------- #
# D3 — L2-merged surfaces persist as aliases across save/reload
# --------------------------------------------------------------------------- #
def test_entity_alias_persists_across_save_reload():
    path = tmp_store()
    store = GraphStore.open(path, cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    a = canon.resolve_entity("Alan Turing", EntityType.PERSON)
    # embedding-synonymy (L2) hard merge records the merged surface as an alias
    assert canon.resolve_entity("Mr. Alan Turing", EntityType.PERSON) == a
    assert "mr. alan turing" in store.get_node(a).aliases
    store.save()
    store2 = GraphStore.open(path, cfg())
    canon2 = Canonicalizer(store2, get_embedder(cfg()), cfg())
    # the reloaded canonicalizer L1-hits the same anchor via the persisted alias
    assert canon2.resolve_entity("Mr. Alan Turing", EntityType.PERSON) == a
    assert canon2.resolve_entity("Alan Turing", EntityType.PERSON) == a


# --------------------------------------------------------------------------- #
# D4 — trailing argument-structure marker keeps predicates distinct
# --------------------------------------------------------------------------- #
def test_works_at_and_works_with_are_distinct_relation_nodes():
    store, canon = fresh_canon()
    at = canon.resolve_relation("works_at")
    with_ = canon.resolve_relation("works_with")
    assert at != with_
    assert relation_merge_vetoed("works_at", "works_with")       # veto guards L3 too
    # friend_of vs friends_with: distinct nodes, BOTH symmetric (fork lexicon)
    of = canon.resolve_relation("friend_of")
    fw = canon.resolve_relation("friends_with")
    assert of != fw
    assert store.get_node(of).symmetric and store.get_node(fw).symmetric
    # tense variants of the SAME predicate still collapse
    assert canon.resolve_relation("worked_with") == with_


# --------------------------------------------------------------------------- #
# E2 — retracted end-to-end: absent from current AND as-of views
# --------------------------------------------------------------------------- #
def test_retract_removes_fact_from_current_and_as_of_views():
    s = temporal_store()
    assert apply_fact(s, src="e_me", dst="e_ann", rel_tag="rel_knows",
                      status="asserted", at="2021", episode_id="ep1") == "open"
    assert apply_fact(s, src="e_me", dst="e_ann", rel_tag="rel_knows",
                      status="retracted", at="2023", episode_id="ep2") == "retract"
    datas = [d for _u, _v, d in s.all_edges() if d.get("rel_tag") == "rel_knows"]
    assert len(datas) == 1 and datas[0]["belief"] == Belief.RETRACTED.value
    assert datas[0]["retracted_at"] == "2023"
    assert datas[0]["retracted_by_episode"] == "ep2"
    # never-true: absent from the current view AND from every as-of-T view,
    # including instants inside the formerly-believed window
    assert not fact_active(datas[0], None)
    assert not fact_active(datas[0], "2022")
    assert list(s.find_facts("e_me")) == []                      # excluded from reads
    assert FactIndex(s).history(["e_me"]) == []                  # …and from history
    assert s.stats()["facts"]["retracted"] == 1
    # a retraction is not a close: a later re-assert opens a FRESH edge
    assert apply_fact(s, src="e_me", dst="e_ann", rel_tag="rel_knows",
                      status="asserted", at="2024", episode_id="ep3") == "open"
    live = [d for _u, _v, d in s.all_edges()
            if d.get("rel_tag") == "rel_knows" and fact_active(d, None)]
    assert len(live) == 1 and live[0]["valid_at"] == "2024"


def test_retract_first_records_never_active_edge():
    s = temporal_store()
    assert apply_fact(s, src="e_me", dst="e_bob", rel_tag="rel_knows",
                      status="retracted", at="2022", episode_id="ep9") == "retract_new"
    d = [d for _u, _v, d in s.all_edges() if d.get("rel_tag") == "rel_knows"][0]
    assert d["belief"] == Belief.RETRACTED.value
    assert d["retracted_by_episode"] == "ep9"
    assert not fact_active(d, None) and not fact_active(d, "2022")


# --------------------------------------------------------------------------- #
# E1 — dispute gating: a weak overturn records a dispute, never mutates
# --------------------------------------------------------------------------- #
def test_dispute_gating_low_confidence_close_keeps_edge_open():
    s = temporal_store()
    apply_fact(s, src="e_me", dst="e_tor", rel_tag="rel_lives", status="asserted",
               at="2020", confidence=0.9, episode_id="ep1")
    # 0.5 < 0.9 - dispute_confidence_margin (0.3 default) → gated
    assert apply_fact(s, src="e_me", dst="e_tor", rel_tag="rel_lives", status="ended",
                      at="2023", confidence=0.5, episode_id="ep2") == "dispute"
    d = next(s.find_facts("e_me", "e_tor", "rel_lives"))[2]
    assert d["invalid_at"] == ""                                 # still open, still believed
    assert d["disputed_by"] == [{"episode": "ep2", "confidence": 0.5,
                                 "at": "2023", "action": "close"}]
    apply_fact(s, src="e_me", dst="e_tor", rel_tag="rel_lives", status="ended",
               at="2023", confidence=0.5, episode_id="ep2")      # replay dedups
    assert len(d["disputed_by"]) == 1
    # a weak retract is gated the same way
    assert apply_fact(s, src="e_me", dst="e_tor", rel_tag="rel_lives",
                      status="retracted", at="2023", confidence=0.4,
                      episode_id="ep3") == "dispute"
    assert d["belief"] == Belief.ASSERTED.value
    # an equal-strength close fires normally and stamps its provenance (E4)
    assert apply_fact(s, src="e_me", dst="e_tor", rel_tag="rel_lives", status="ended",
                      at="2025", confidence=0.9, episode_id="ep5") == "close"
    d = next(s.find_facts("e_me", "e_tor", "rel_lives"))[2]
    assert d["invalid_at"] == "2025" and d["closed_at"] == "2025"
    assert d["closed_by_episode"] == "ep5"


# --------------------------------------------------------------------------- #
# E3 — symmetric-functional supersede across stored orientations
# --------------------------------------------------------------------------- #
def test_symmetric_functional_supersede_across_orientations():
    s = temporal_store()
    apply_fact(s, src="e_me", dst="e_ann", rel_tag="rel_spouse", status="asserted",
               at="2015", episode_id="ep1")       # symmetric pinning stores (e_ann, e_me)
    # the new marriage arrives in the OTHER orientation and must still close the old one
    assert apply_fact(s, src="e_bob", dst="e_me", rel_tag="rel_spouse",
                      status="asserted", at="2020", episode_id="ep2") == "open"
    old = next(s.find_facts("e_ann", "e_me", "rel_spouse"))[2]
    assert old["invalid_at"] == "2020" and old["closed_at"] == "2020"
    assert old["closed_by_episode"] == "ep2"
    new = next(s.find_facts("e_bob", "e_me", "rel_spouse"))[2]
    assert new["invalid_at"] == ""


# --------------------------------------------------------------------------- #
# C3 — once_/past_ are NOT termination prefixes (false-positive closers)
# --------------------------------------------------------------------------- #
def test_termination_regex_ignores_once_and_past_prefixes():
    def rel(label):
        ext = _parse_tool_payload({
            "entities": [{"name": "A", "type": "person"},
                         {"name": "B", "type": "person"}],
            "tags": [],
            "relations": [{"source": "A", "target": "B", "labels": [label]}],
        })
        return ext.relations[0]

    assert rel("once_met").status == "asserted"                  # happened ≠ ended
    assert rel("once_met").labels == ["once_met"]
    assert rel("past_project").status == "asserted"
    # the unambiguous former-markers still close, folding onto the base predicate
    assert rel("no_longer_works_with").status == "ended"
    assert rel("no_longer_works_with").labels == ["works_with"]
    assert rel("former_colleague").status == "ended"
    assert rel("ex-coworker").status == "ended"


# --------------------------------------------------------------------------- #
# C1 — payload back-compat: fork's "concepts" key + term/preference types
# --------------------------------------------------------------------------- #
def test_concepts_payload_key_backcompat():
    ext = _parse_tool_payload({"entities": [], "concepts": ["alpha", "beta"]})
    assert ext.tags == ["alpha", "beta"]
    # "tags" still wins when present
    ext = _parse_tool_payload({"entities": [], "tags": ["gamma"],
                               "concepts": ["alpha"]})
    assert ext.tags == ["gamma"]
    # C5: process/meta tags are stoplisted case-insensitively
    ext = _parse_tool_payload({"entities": [], "concepts": ["alpha", "Shipped", "test"]})
    assert ext.tags == ["alpha"]


def test_term_and_preference_entity_type_aliases():
    ext = _parse_tool_payload({"entities": [
        {"name": "recursion", "type": "term"},          # fork's name for CONCEPT
        {"name": "likes jazz", "type": "preference"},   # fork enum member (B1)
        {"name": "mystery", "type": "wibble"},          # unknown → OTHER
    ], "tags": []})
    by_name = {e.name: e for e in ext.entities}
    assert by_name["recursion"].type == EntityType.CONCEPT
    assert by_name["likes jazz"].type == EntityType.PREFERENCE
    assert by_name["mystery"].type == EntityType.OTHER
    # category is always populated on parsed entities (type-derived fallback)
    assert all(e.category == EntityCategory.THING for e in ext.entities)


# --------------------------------------------------------------------------- #
# F2 — entity_category stamped at ingest, THING upgrades on a person mention
# --------------------------------------------------------------------------- #
def test_category_stamped_at_ingest_and_thing_upgrade():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.extractor = ScriptedExtractor({
        "robin joined.": Extraction(
            entities=[ExtractedEntity("Robin", EntityType.OTHER)], tags=["team"]),
        "robin is a person.": Extraction(
            entities=[ExtractedEntity("Robin", EntityType.PERSON)], tags=["team"]),
    })
    g.ingest([CorpusItem(id="a", modality="text", source_ref="a",
                         text="Robin joined.", created_at="2024-01-01T00:00:00")])
    anchor = next(n for n in g.store.nodes_of_type(NodeType.ENTITY)
                  if n.name == "Robin")
    assert anchor.entity_category == EntityCategory.THING        # OTHER → THING glyph
    g.ingest([CorpusItem(id="b", modality="text", source_ref="b",
                         text="Robin is a person.", created_at="2024-01-02T00:00:00")])
    anchor = g.store.get_node(anchor.id)
    assert anchor.entity_category == EntityCategory.PERSON       # upgraded in place
    assert anchor.entity_type == EntityType.PERSON


# --------------------------------------------------------------------------- #
# F3 — salted content hash: distinct captures ingest, true re-ingests skip
# --------------------------------------------------------------------------- #
def test_salted_hash_dedup_distinct_ids_vs_reingest():
    g = KnowledgeGraph.open(tmp_store(), cfg())
    g.extractor = ScriptedExtractor({})
    same_text = "Coffee with Sam."
    rep = g.ingest([
        CorpusItem(id="cap1", modality="text", source_ref="s1", text=same_text,
                   created_at="2024-03-01T09:00:00"),
        CorpusItem(id="cap2", modality="text", source_ref="s2", text=same_text,
                   created_at="2024-03-01T09:00:00"),
    ])
    # byte-identical text under DISTINCT ids = two deliberate captures → two episodes
    assert rep.ingested == 2 and rep.skipped == 0
    assert len(g.store.nodes_of_type(NodeType.EPISODE)) == 2
    # a true re-ingest (same id, same content) still skips
    rep2 = g.ingest([CorpusItem(id="cap1", modality="text", source_ref="s1",
                                text=same_text, created_at="2024-03-01T09:00:00")])
    assert rep2.ingested == 0 and rep2.skipped == 1
    assert len(g.store.nodes_of_type(NodeType.EPISODE)) == 2


# --------------------------------------------------------------------------- #
# A1 — provider auto-detect ordering (mocked probes, env-manipulated)
# --------------------------------------------------------------------------- #
def test_provider_autodetect_ordering(monkeypatch):
    from kg import llm_client

    monkeypatch.delenv("KG_LLM", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def probes(codex: bool, claude: bool):
        monkeypatch.setattr(llm_client, "_codex_login_status",
                            lambda: (codex, "mock"))
        monkeypatch.setattr(llm_client, "_claude_login_status",
                            lambda: (claude, "mock"))

    # subscription CLIs first: codex outranks everything …
    probes(codex=True, claude=True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    assert llm_client.detect_provider() == "codex"
    # … then claude …
    probes(codex=False, claude=True)
    assert llm_client.detect_provider() == "claude"
    # … then the API keys, anthropic before openai …
    probes(codex=False, claude=False)
    assert llm_client.detect_provider() == "anthropic"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert llm_client.detect_provider() == "openai"
    # … and nothing live → none
    monkeypatch.delenv("OPENAI_API_KEY")
    assert llm_client.detect_provider() == "none"
    # an explicit KG_LLM always wins over the probe chain
    probes(codex=True, claude=True)
    monkeypatch.setenv("KG_LLM", "openai")
    assert llm_client.current_provider()["kind"] == "openai"
