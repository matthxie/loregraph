"""Temporal evolution tests (docs/TEMPORAL.md) — the Becky / Alex timeline.

Fully offline/deterministic: the synthetic stream + ScriptedExtractor feed the bi-temporal
ingest logic clean facts, so the thing under test is the graph's evolution (open / confirm /
close / supersede / backfill) and the as-of-T retrieval over it. Run: python -m pytest -q
"""
from __future__ import annotations

import os
import tempfile

import pytest

import kg.graph as kg_graph
from kg import Config, KnowledgeGraph
from kg.canonicalize import Canonicalizer
from kg.corpus import CorpusItem
from kg.embedders import SentenceTransformerEmbedder, get_embedder
from kg.extractors import Extraction, ExtractedEntity, ScriptedExtractor
from kg.models import EdgeType, EntityType, NodeType, entity_node
from kg.store import GraphStore, fact_active
from kg.synthetic import becky_stream
from kg.temporal import apply_fact


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    """Keep the whole module deterministic + free + key-independent.

    The library is LIVE-ONLY: ``kg`` auto-loads a project-root ``.env`` on import and
    ``KnowledgeGraph.__init__`` eagerly builds a (live) HaikuExtractor via get_extractor,
    which RAISES without ANTHROPIC_API_KEY. These temporal tests never touch the LLM — the
    facts come from the synthetic stream's ScriptedExtractor and the assertions are about
    the graph's bi-temporal evolution. So we (a) drop any key the .env may have injected and
    (b) make the graph build a ScriptedExtractor instead of the live one. becky_graph() then
    overrides g.extractor with the real Becky table anyway, so extraction stays deterministic
    and no Anthropic call is ever made."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(kg_graph, "get_extractor",
                        lambda config: ScriptedExtractor({}))


def cfg() -> Config:
    c = Config.default()
    c.embedder = "st"   # real local bge — deterministic, free, no key, no network once cached
    return c


def becky_graph(extra_silence: bool = False) -> KnowledgeGraph:
    g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), cfg())
    items, table = becky_stream()
    if extra_silence:
        # an episode that mentions Becky but asserts NO relationship — the open-world rule
        # says this must NOT close any of her existing facts.
        txt = "Becky enjoys hiking on weekends."
        table[txt] = Extraction(entities=[ExtractedEntity("Becky", EntityType.PERSON)],
                                tags=["hobby"], relations=[])
        items.append(CorpusItem(id="becky05", modality="text", source_ref="synthetic/becky05",
                                title="Becky's hobby", text=txt,
                                created_at="2025-01-01T00:00:00+00:00"))
    g.extractor = ScriptedExtractor(table)
    g.ingest(items)
    return g


def becky_id(g) -> str:
    return next(n.id for n in g.store.nodes_of_type(NodeType.ENTITY) if n.name == "Becky")


def active_facts(g, as_of=None) -> set:
    """Becky's active (current or as-of-T) facts as {(predicate, other_entity)}."""
    becky = becky_id(g)
    out = set()
    for nbr, d in g.store.neighbors(becky, etypes={EdgeType.RELATED_TO}, direction="both"):
        if fact_active(d, as_of):
            rel = g.store.get_node(d["rel_tag"]).name
            other = g.store.get_node(nbr).name
            out.add((rel, other))
    return out


# --------------------------------------------------------------------------- #
# end-to-end timeline
# --------------------------------------------------------------------------- #
def test_current_view_reflects_latest_state():
    g = becky_graph()
    assert active_facts(g, None) == {
        ("lives_in", "Berlin"), ("works_with", "Dana"), ("employed_by", "Globex")}


def test_supersede_and_close_remove_stale_facts():
    g = becky_graph()
    now = active_facts(g, None)
    # functional supersession dropped the old single-valued values
    assert ("lives_in", "Toronto") not in now      # superseded by Berlin (2023)
    assert ("employed_by", "Acme Corp") not in now  # superseded by Globex (2024)
    # termination closed the relationship
    assert ("works_with", "Alex") not in now        # ended (2024)


def test_asof_recovers_history():
    g = becky_graph()
    assert active_facts(g, "2022") == {
        ("lives_in", "Toronto"), ("works_with", "Alex"), ("employed_by", "Acme Corp")}


def test_asof_midpoint_is_distinct_from_now_and_start():
    g = becky_graph()
    # mid-2023: already in Berlin, but still works_with Alex and at Acme (both end in 2024)
    mid = active_facts(g, "2023-08")
    assert ("lives_in", "Berlin") in mid
    assert ("works_with", "Alex") in mid
    assert ("lives_in", "Toronto") not in mid


def test_open_world_silence_preserves_facts():
    """A later episode that mentions Becky but states no relationship must NOT close her
    open facts — closure requires positive evidence."""
    g = becky_graph(extra_silence=True)
    now = active_facts(g, None)
    assert ("works_with", "Dana") in now and ("lives_in", "Berlin") in now


def test_one_stable_entity_node_across_evolution():
    g = becky_graph()
    beckys = [n for n in g.store.nodes_of_type(NodeType.ENTITY) if n.name == "Becky"]
    assert len(beckys) == 1  # identity is the anchor; only edges evolve


# --------------------------------------------------------------------------- #
# apply_fact unit logic
# --------------------------------------------------------------------------- #
def _mini():
    store = GraphStore(cfg())
    canon = Canonicalizer(store, get_embedder(cfg()), cfg())
    for nid, name in [("e_becky", "Becky"), ("e_tor", "Toronto"),
                      ("e_ber", "Berlin"), ("e_alex", "Alex")]:
        store.add_node(entity_node(nid, name=name, etype=EntityType.PERSON, ts="t"))
    return store, canon


def test_apply_fact_open_confirm_supersede_close():
    store, canon = _mini()
    lives = canon.resolve_relation("lives_in")      # functional
    works = canon.resolve_relation("works_with")    # symmetric
    assert apply_fact(store, src="e_becky", dst="e_tor", rel_tag=lives,
                      status="asserted", at="2021") == "open"
    assert apply_fact(store, src="e_becky", dst="e_tor", rel_tag=lives,
                      status="asserted", at="2021-07") == "confirm"
    # functional new value supersedes: Toronto closes, Berlin opens
    assert apply_fact(store, src="e_becky", dst="e_ber", rel_tag=lives,
                      status="asserted", at="2023") == "open"
    assert next(store.find_facts("e_becky", "e_tor", lives))[2]["invalid_at"] == "2023"
    assert next(store.find_facts("e_becky", "e_ber", lives))[2]["invalid_at"] == ""
    # termination closes
    apply_fact(store, src="e_becky", dst="e_alex", rel_tag=works, status="asserted", at="2021")
    assert apply_fact(store, src="e_becky", dst="e_alex", rel_tag=works,
                      status="ended", at="2024") == "close"


def test_apply_fact_backfill_order_independence():
    store, canon = _mini()
    works = canon.resolve_relation("works_with")    # symmetric → consistent orientation
    # learn the END first → a closed edge with unknown start (never fabricated)
    assert apply_fact(store, src="e_becky", dst="e_alex", rel_tag=works,
                      status="ended", at="2024", valid_to="2024") == "open_closed"
    # then learn the START → backfill the same edge, not a duplicate
    assert apply_fact(store, src="e_becky", dst="e_alex", rel_tag=works,
                      status="asserted", at="2021", valid_from="2021") == "backfill"
    facts = list(store.find_facts("e_alex", "e_becky", works))  # symmetric pinned orientation
    assert len(facts) == 1
    assert facts[0][2]["valid_at"] == "2021" and facts[0][2]["invalid_at"] == "2024"


def test_symmetric_predicate_stored_once():
    store, canon = _mini()
    works = canon.resolve_relation("works_with")
    apply_fact(store, src="e_becky", dst="e_alex", rel_tag=works, status="asserted", at="2021")
    apply_fact(store, src="e_alex", dst="e_becky", rel_tag=works, status="asserted", at="2021")
    # both orientations collapse to one fact edge
    rel_edges = [(u, v) for u, v, d in store.all_edges()
                 if d["etype"] == EdgeType.RELATED_TO.value]
    assert len(rel_edges) == 1


# --------------------------------------------------------------------------- #
# live-only backend contract (replaces the old offline-backend coverage that this
# file used to lean on via cfg(); the temporal layer above runs over the real local
# bge embedder, and the live extractor must refuse to construct without a key)
# --------------------------------------------------------------------------- #
def test_get_embedder_is_sentence_transformer():
    """The selectable HashingEmbedder was removed: get_embedder ALWAYS returns the local
    semantic embedder now, which is what these temporal tests' retrieval seeds run on."""
    emb = get_embedder(cfg())
    assert isinstance(emb, SentenceTransformerEmbedder)
    assert emb.name.startswith("st:")


def test_get_extractor_requires_key(monkeypatch):
    """Extraction is live-only: with no ANTHROPIC_API_KEY, get_extractor RAISES rather than
    falling back to the deleted offline heuristic. (The temporal suite sidesteps this by
    constructing a ScriptedExtractor directly — see the _no_live_llm fixture.)"""
    from kg.extractors import get_extractor

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        get_extractor(cfg())
