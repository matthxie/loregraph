"""Self-anchor PPR hub guard tests (kg.config.self_guard).

Deterministic + free: builds a real self-hub graph by ingesting the synthetic
personal_stream with a ScriptedExtractor (no key), then asserts how each guard mode
shapes `projected_graph`. Covers the contract the guard promises:

  none     — self present, incident edges intact (byte-for-byte unchanged default)
  exclude  — self node AND its incident edges absent from the projection
  cap      — self present but its incident-edge mass throttled vs none
  seed     — structurally identical to none (the seed-skip is a PPR-time choice)

Run: python -m pytest tests/test_self_guard.py -q
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import replace

from kg import Config
from kg.graph import KnowledgeGraph
from kg.extractors import ScriptedExtractor
from kg.models import SELF_ENTITY_ID
from kg.retrieval import projected_graph
from kg.synthetic import personal_stream


def _cfg(**over) -> Config:
    c = Config.default()
    c.embedder = "st"
    c.self_entity = True
    return replace(c, **over) if over else c


def _build(monkeypatch) -> KnowledgeGraph:
    items, table = personal_stream()
    scripted = ScriptedExtractor(table)
    monkeypatch.setattr("kg.graph.get_extractor", lambda cfg: scripted)
    path = os.path.join(tempfile.mkdtemp(), "kg.db")
    g = KnowledgeGraph.open(path, _cfg())
    g.ingest(items)
    assert g.store.has_node(SELF_ENTITY_ID), "self anchor should exist after personal ingest"
    return g


def _self_incident_weights(G):
    return [d["weight"] for _u, _v, d in G.edges(SELF_ENTITY_ID, data=True)] \
        if SELF_ENTITY_ID in G else []


def test_none_keeps_self_as_hub(monkeypatch):
    g = _build(monkeypatch)
    G = projected_graph(g.store, replace(g.config, self_guard="none"))
    assert SELF_ENTITY_ID in G
    assert G.degree(SELF_ENTITY_ID) > 0, "self should be a hub in the unguarded projection"


def test_default_is_byte_for_byte_none(monkeypatch):
    g = _build(monkeypatch)
    G_default = projected_graph(g.store, g.config)                       # default self_guard="none"
    G_none = projected_graph(g.store, replace(g.config, self_guard="none"))
    assert set(G_default.edges()) == set(G_none.edges())
    assert set(G_default.nodes()) == set(G_none.nodes())


def test_exclude_drops_self_and_its_edges(monkeypatch):
    g = _build(monkeypatch)
    G = projected_graph(g.store, replace(g.config, self_guard="exclude"))
    assert SELF_ENTITY_ID not in G, "exclude must drop the self node"
    for u, v in G.edges():
        assert u != SELF_ENTITY_ID and v != SELF_ENTITY_ID, "no edge may touch self under exclude"


def test_cap_throttles_self_mass(monkeypatch):
    g = _build(monkeypatch)
    cap = 0.05
    G_none = projected_graph(g.store, replace(g.config, self_guard="none"))
    G_cap = projected_graph(g.store, replace(g.config, self_guard="cap", self_guard_cap=cap))
    assert SELF_ENTITY_ID in G_cap, "cap keeps the self node"
    w_none = sum(_self_incident_weights(G_none))
    w_cap = sum(_self_incident_weights(G_cap))
    if w_none > len(_self_incident_weights(G_none)) * cap:    # only if there was mass to cap
        assert w_cap < w_none, "cap must reduce total self-incident weight"


def test_seed_keeps_projection_like_none(monkeypatch):
    g = _build(monkeypatch)
    G_none = projected_graph(g.store, replace(g.config, self_guard="none"))
    G_seed = projected_graph(g.store, replace(g.config, self_guard="seed"))
    # seed only changes PPR personalization, not the projection structure
    assert set(G_seed.edges()) == set(G_none.edges())
    assert SELF_ENTITY_ID in G_seed
