"""Round 7b — the FACT LANE (config.fact_lane): statement-granularity retrieval feeding
the episode pipeline. Scores the query against the kind="fact" vectors (Round 7a), maps the
top hits back to their provenance chunks + endpoint entities, and merges those ADDITIVELY
into the seed set so a fact's asserting chunk enters the PPR pool because its CLAIM matched —
the fix for the dilution class (docs/OFFLINE_EVAL.md Round 7b, the 06f04340 trace).

The contracts under test:
  * fact_provenance maps a statement/aggregate vector id back to episodes + entities;
  * the lane pulls an OFF-TOPIC needle (a fact stated in passing) that the episode lane
    misses into the seed set / pool / context, marked [matched] in FACTS;
  * ADDITIVE-ONLY — every episode-lane seed keeps its EXACT score, total fact-lane mass is
    capped at fact_lane_weight × the episode-lane mass;
  * knob OFF ⇒ seeds/pool/context byte-identical (the lane never runs).

Offline/deterministic: the real local bge embedder (so fact-vector cosine is meaningful) and
a ScriptedExtractor (no LLM). Mirrors tests/test_fact_vectors.py.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import replace

import pytest

import kg.graph as kg_graph
from kg import Config, KnowledgeGraph
from kg.corpus import CorpusItem
from kg.extractors import (ExtractedEntity, ExtractedRelation, Extraction,
                           ScriptedExtractor)
from kg.fact_vectors import FACT_KIND, fact_provenance
from kg.models import EntityType, NodeType, Provenance
from kg.rag import ContextBuilder
from kg.retrieval import HybridRetriever, PPRRetriever, Seeder


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(kg_graph, "get_extractor", lambda config: ScriptedExtractor({}))


def _E(name):
    return ExtractedEntity(
        name=name, type=EntityType.PERSON if name == "me" else EntityType.CONCEPT)


def _R(src, tgt, label, at=""):
    return ExtractedRelation(source=src, target=tgt, labels=[label],
                             provenance=Provenance.EXTRACTED, confidence=0.95,
                             status="asserted", valid_from=at)


# The NEEDLE: pure flight-logistics text (shares no vocabulary with "blood type") whose
# scripted extraction asserts a blood-type fact in passing — exactly the passing-mention
# geometry Round 5 measured (the sentence never says the query word). Fourteen distractors
# push the corpus past seed_k so an off-topic chunk falls out of the episode lane.
_NEEDLE = ("flight", "2024-03-01",
           "Sorted out the trip logistics today: booked the United flight to Chicago, "
           "reserved an aisle seat, and printed the boarding passes for the whole family.",
           [("me", "O-negative", "has_blood_type")], ["me", "O-negative"])
_FILL = [
    ("cafe", "coffee shop Cafe Vivace mornings", [("me", "Cafe Vivace", "frequents")], ["me", "Cafe Vivace"]),
    ("gym", "Joined Flex Fitness the gym", [("me", "Flex Fitness", "member_of")], ["me", "Flex Fitness"]),
    ("pet", "adopted a cat named Luna", [("me", "Luna", "has_pet")], ["me", "Luna"]),
    ("car", "bought a Subaru Outback", [("me", "Subaru Outback", "drives")], ["me", "Subaru Outback"]),
    ("gtr", "practicing guitar in the evenings", [("me", "guitar", "plays")], ["me", "guitar"]),
    ("spn", "picking up Spanish lessons", [("me", "Spanish", "speaks")], ["me", "Spanish"]),
    ("bank", "I bank with Chase", [("me", "Chase", "banks_with")], ["me", "Chase"]),
    ("mac", "use a MacBook for work", [("me", "MacBook", "uses")], ["me", "MacBook"]),
    ("team", "Cheering for the Sounders", [("me", "Sounders", "supports")], ["me", "Sounders"]),
    ("city", "I grew up in Portland", [("me", "Portland", "grew_up_in")], ["me", "Portland"]),
    ("sf", "I mostly read science fiction", [("me", "science fiction", "reads")], ["me", "science fiction"]),
    ("plant", "collecting houseplants now", [("me", "houseplants", "collects")], ["me", "houseplants"]),
    ("desk", "bought a standing desk", [("me", "standing desk", "owns")], ["me", "standing desk"]),
    ("kom", "I brew my own kombucha", [("me", "kombucha", "brews")], ["me", "kombucha"]),
]
_QUERY = "what is my blood type?"


def _base_cfg(**over) -> Config:
    c = Config.default()
    c.embedder = "st"
    c.self_entity = True
    c.self_name = "me"
    c.event_facts = True
    c.fact_vectors = True
    for k, v in over.items():
        setattr(c, k, v)
    return c


@pytest.fixture(scope="module")
def needle_graph():
    """A 15-episode store (needle + 14 distractors) ingested with fact_vectors on, so every
    believed fact has a kind='fact' vector. Built once (real bge is slow) and reused."""
    for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.pop(_k, None)
    kg_graph.get_extractor = lambda config: ScriptedExtractor({})
    cfg = _base_cfg()
    g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), cfg)
    rows = [_NEEDLE] + [(eid, "2020-01-01", txt, rels, ents)
                        for eid, txt, rels, ents in _FILL]
    items, table = [], {}
    for eid, day, text, rels, ents in rows:
        items.append(CorpusItem(id=eid, modality="text", source_ref=f"synthetic/{eid}",
                                title=eid, text=text, created_at=f"{day}T12:00:00+00:00"))
        table[text] = Extraction(entities=[_E(n) for n in ents], tags=["personal"],
                                 relations=[_R(*r) for r in rels])
    g.extractor = ScriptedExtractor(table)
    g.ingest(items)
    g.save()
    return g


# --------------------------------------------------------------------------- #
# fact_provenance: vector id -> provenance episodes + endpoint entities
# --------------------------------------------------------------------------- #
def test_fact_provenance_resolves_statement(needle_graph):
    g = needle_graph
    ids = g.store.vectors.ids(FACT_KIND)
    # find the blood-type statement id via its surface
    from kg.fact_vectors import current_surfaces
    stmt, _agg = current_surfaces(g.store)
    blood_id = next(i for i, s in stmt.items() if s == "me has_blood_type O-negative")
    prov = fact_provenance(g.store, {blood_id})
    rec = prov[blood_id]
    assert rec["stmt_surface"] == "me has_blood_type O-negative"
    assert "ep_flight" in rec["episodes"]           # the asserting chunk is the provenance
    # both endpoints resolve to entity nodes
    assert len(rec["entities"]) == 2
    assert all(g.store.get_node(e).ntype == NodeType.ENTITY for e in rec["entities"])
    assert ids  # sanity: the store really has fact vectors


def test_fact_provenance_aggregate_unions_group():
    """An aggregate hit resolves to its whole (src,rel,dst) group's provenance episodes."""
    from kg.canonicalize import Canonicalizer
    from kg.embedders import get_embedder
    from kg.fact_vectors import current_surfaces, sync_fact_vectors
    from kg.models import entity_node
    from kg.store import GraphStore
    from kg.temporal import apply_fact
    cfg = _base_cfg()
    path = os.path.join(tempfile.mkdtemp(), "kg.db")
    store = GraphStore(cfg, path=path)
    store._init_db()
    canon = Canonicalizer(store, get_embedder(cfg), cfg)
    for nid, name in [("e_me", "me"), ("e_park", "the park")]:
        store.add_node(entity_node(nid, name=name, etype=EntityType.CONCEPT, ts="t"))
    went = canon.resolve_relation("went_to")
    for d, ep in [("2025-01-05", "ep_p1"), ("2025-02-02", "ep_p2"), ("2025-03-10", "ep_p3")]:
        apply_fact(store, src="e_me", dst="e_park", rel_tag=went, status="asserted",
                   at=d, episode_id=ep)
    sync_fact_vectors(store, get_embedder(cfg), prune=True)
    _stmt, agg = current_surfaces(store)
    agg_id = next(iter(agg))
    prov = fact_provenance(store, {agg_id})
    assert prov[agg_id]["episodes"] == {"ep_p1", "ep_p2", "ep_p3"}   # whole group
    assert prov[agg_id]["stmt_surface"] == "me went_to the park"


# --------------------------------------------------------------------------- #
# fact_seed: the off-topic needle the episode lane misses
# --------------------------------------------------------------------------- #
def test_fact_seed_catches_needle_episode_lane_misses(needle_graph):
    g = needle_graph
    cfg = _base_cfg()
    sdr = Seeder(g.store, g.embedder, g.canon, cfg)
    ep_seeds = sdr.seed(_QUERY)
    raw, surfaces, hits = sdr.fact_seed(_QUERY)
    # the episode lane does NOT reach the off-topic flight chunk (needle geometry)
    assert "ep_flight" not in ep_seeds
    # the fact lane's top hit is the blood-type statement, mapped to the flight chunk
    assert surfaces[0] == "me has_blood_type O-negative"
    assert "ep_flight" in raw and raw["ep_flight"] > 0
    assert hits and hits[0][0].startswith("fact:")


def test_fact_seed_empty_without_vectors():
    """No fact vectors present (un-backfilled store) ⇒ the lane no-ops cleanly."""
    cfg = _base_cfg(fact_vectors=False)
    g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), cfg)
    items = [CorpusItem(id="x", modality="text", source_ref="s", title="x",
                        text="I adopted a cat named Luna.", created_at="2020-01-01T00:00:00+00:00")]
    g.extractor = ScriptedExtractor(
        {items[0].text: Extraction(entities=[_E("me"), _E("Luna")], tags=["p"],
                                   relations=[_R("me", "Luna", "has_pet")])})
    g.ingest(items)
    raw, surfaces, hits = Seeder(g.store, g.embedder, g.canon, cfg).fact_seed("cat name?")
    assert raw == {} and surfaces == [] and hits == []


# --------------------------------------------------------------------------- #
# ADDITIVE-ONLY: episode-lane seeds and their scores are provably unchanged
# --------------------------------------------------------------------------- #
def test_merge_is_additive_episode_seeds_unchanged(needle_graph, monkeypatch):
    """The core Round-7b invariant: merging the fact lane adds only NEW nodes and never alters
    an episode-lane seed's score — even when the fact lane names a node the episode lane
    already seeded (that node keeps the EPISODE-lane score, not the fact score)."""
    g = needle_graph
    cfg = _base_cfg(fact_lane=True, fact_lane_weight=0.5)
    ppr = PPRRetriever(g.store, g.embedder, g.canon, cfg)
    ep_seeds = dict(ppr.seeder.seed(_QUERY))          # the pure episode lane

    # fact lane returns a node the episode lane ALSO has (an ep_seeds key) + a brand-new one
    shared = next(iter(ep_seeds))
    monkeypatch.setattr(ppr.seeder, "fact_seed",
                        lambda q: ({shared: 0.99, "ep_flight": 0.8, "ent_new": 0.7},
                                   ["me has_blood_type O-negative"], [("fact:z", 0.8)]))
    merged, matched = ppr._merge_fact_lane(_QUERY, dict(ep_seeds))

    for nid, sc in ep_seeds.items():
        assert merged[nid] == sc                      # byte-identical, incl. the `shared` key
    assert "ep_flight" in merged and "ent_new" in merged    # new nodes added
    assert set(merged) - set(ep_seeds) == {"ep_flight", "ent_new"}
    assert matched["surfaces"] == ["me has_blood_type O-negative"]


def test_mass_cap_bounds_fact_lane_contribution(needle_graph, monkeypatch):
    g = needle_graph
    cfg = _base_cfg(fact_lane=True, fact_lane_weight=0.5)
    ppr = PPRRetriever(g.store, g.embedder, g.canon, cfg)
    ep_seeds = {"ep_a": 1.0, "ent_b": 0.6}            # episode mass = 1.6, cap = 0.8
    # raw new mass 0.8 + 0.7 = 1.5 > cap → must be scaled down to exactly the cap
    monkeypatch.setattr(ppr.seeder, "fact_seed",
                        lambda q: ({"ep_new": 0.8, "ent_new": 0.7}, ["s"], [("fact:z", 0.8)]))
    merged, _ = ppr._merge_fact_lane(_QUERY, dict(ep_seeds))
    added = sum(merged[n] for n in merged if n not in ep_seeds)
    assert added == pytest.approx(0.5 * 1.6, rel=1e-9)   # Σ fact mass ≤ weight × episode mass
    assert merged["ep_a"] == 1.0 and merged["ent_b"] == 0.6


# --------------------------------------------------------------------------- #
# knob OFF ⇒ byte-identical (the lane never runs)
# --------------------------------------------------------------------------- #
def test_knob_off_seeds_byte_identical(needle_graph):
    g = needle_graph
    off = _base_cfg(fact_lane=False)
    ppr = PPRRetriever(g.store, g.embedder, g.canon, off)
    res = ppr.retrieve(_QUERY, k=8)
    pure = ppr.seeder.seed(_QUERY)
    assert res.seed_scores == pure                    # fact lane left seeds untouched
    assert res.fact_matched == {}                     # nothing plumbed downstream


def test_knob_off_context_has_no_matched_mark(needle_graph):
    g = needle_graph
    off = _base_cfg(fact_lane=False, rag_provenance_promote=True,
                    rag_parent_expand=2, rag_chunks_per_source=2)
    res = HybridRetriever(g.store, g.embedder, g.canon, off).retrieve(_QUERY, k=8)
    _ctx, _facts, blob = ContextBuilder(g.store, off).build(res)
    assert "[matched]" not in blob


# --------------------------------------------------------------------------- #
# END-TO-END: pool entry + context inclusion + FACTS marking, lane on vs off
# --------------------------------------------------------------------------- #
def _run(g, cfg):
    res = HybridRetriever(g.store, g.embedder, g.canon, cfg).retrieve(_QUERY, k=8)
    ctx, facts, blob = ContextBuilder(g.store, cfg).build(res)
    pool = [e for e, _s in getattr(res, "ppr_pool", [])]
    return res, ctx, blob, pool


def test_lane_seeds_needle_and_marks_fact(needle_graph):
    """Lane on vs off, end to end. In this small single-chunk store PPR diffusion from the
    shared `me` hub reaches every episode, so object-level pool entry is trivial and not the
    signal — the signal is that the fact lane SEEDS the off-topic chunk (its claim matched)
    and the reader sees the matched line flagged. The chunked-store win (provenance-promotion
    seating the needle) is measured in scripts/offline_eval_round7b.py + the mechanism test
    below."""
    g = needle_graph
    common = dict(rag_provenance_promote=True, rag_parent_expand=2, rag_chunks_per_source=2)
    res_off, _ctx_off, blob_off, _pool = _run(g, _base_cfg(fact_lane=False, **common))
    res_on, _ctx_on, blob_on, _pool2 = _run(g, _base_cfg(fact_lane=True, **common))

    # the episode lane never seeds the off-topic chunk; the fact lane does (its CLAIM matched)
    assert "ep_flight" not in res_off.seeds
    assert "ep_flight" in res_on.seeds
    # the matched statement is surfaced and flagged only when the lane is on
    assert "me --has_blood_type--> O-negative" in blob_on and "[matched]" in blob_on
    assert "[matched]" not in blob_off


def test_provenance_promotion_seats_needle_when_sibling_displaceable(needle_graph):
    """The context-side guarantee: given a displaceable expansion sibling (what a chunked
    session always has), a fact-lane needle chunk rides the EXISTING rag_provenance_promote
    path into context via `force_ids` — even though its endpoints ("me", "O-negative") share
    no term with the query, so the ordinary term-overlap gate would skip it."""
    g = needle_graph
    cfg = _base_cfg(fact_lane=True, rag_provenance_promote=True)
    cb = ContextBuilder(g.store, cfg)
    selected = ["ep_cafe"]                     # an originally-selected chunk (never displaced)
    ctx_ids = ["ep_cafe", "ep_gym"]            # ep_gym is expansion-only ⇒ displaceable
    out = cb._promote_provenance(ctx_ids, selected, facts=[], query=_QUERY,
                                 force_ids=["ep_flight"])
    assert out == ["ep_cafe", "ep_flight"]     # needle seated, the sibling gave up its seat
    assert cb.last_retargeted[-1] == {"kind": "provenance_promote",
                                      "displaced": "ep_gym", "promoted": "ep_flight"}
    # gate honoured: with promotion off, force_ids do nothing
    cfg_off = _base_cfg(fact_lane=True, rag_provenance_promote=False)
    assert ContextBuilder(g.store, cfg_off)._promote_provenance(
        ctx_ids, selected, facts=[], query=_QUERY, force_ids=["ep_flight"]) == ctx_ids


def test_matched_mark_only_on_the_matched_line(needle_graph):
    """The [matched] tag lands on the fact the lane hit, not indiscriminately on every FACT."""
    g = needle_graph
    cfg = _base_cfg(fact_lane=True, fact_lane_k=1, rag_provenance_promote=True,
                    rag_parent_expand=2, rag_chunks_per_source=2)
    _res, _ctx, blob, _pool = _run(g, cfg)
    facts_section = blob.split("FACTS currently valid", 1)[1]
    matched_lines = [ln for ln in facts_section.splitlines() if "[matched]" in ln]
    assert matched_lines, "the blood-type line must be marked"
    assert all("has_blood_type" in ln for ln in matched_lines)   # only the hit line
