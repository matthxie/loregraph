"""graph_preview contract tests (PROTOCOL §3.6a): entity/concept roots, predicate
labels on fact edges, and external_connections for off-screen continuation stubs.

Fully offline: ScriptedExtractor feeds known Extractions (no LLM); embeddings use the
local bge model, same policy as the rest of the suite.
"""
from __future__ import annotations

import tempfile

import pytest

from kg.daemon import Daemon
from kg.engine import Engine, NoteInput
from kg.errors import NotFound
from kg.extractors import (Extraction, ExtractedEntity, ExtractedRelation,
                           ScriptedExtractor)
from kg.models import EntityType, NodeType, Provenance

TURING = "Alan Turing worked at Bletchley Park on the Enigma."
PAPER = "Turing wrote a paper about the Enigma."

SCRIPT = {
    TURING: Extraction(
        entities=[ExtractedEntity("Alan Turing", EntityType.PERSON),
                  ExtractedEntity("Bletchley Park", EntityType.PLACE),
                  ExtractedEntity("Enigma", EntityType.CONCEPT)],
        tags=["cryptography"],
        relations=[ExtractedRelation(source="Alan Turing", target="Bletchley Park",
                                     labels=["worked_at"],
                                     provenance=Provenance.EXTRACTED, confidence=0.9)],
    ),
    PAPER: Extraction(
        entities=[ExtractedEntity("Alan Turing", EntityType.PERSON),
                  ExtractedEntity("Enigma", EntityType.CONCEPT)],
        tags=["cryptography"],
    ),
}


@pytest.fixture
def eng():
    e = Engine.open(tempfile.mkdtemp(), {"kind": "mock"})
    e._g.extractor = ScriptedExtractor(SCRIPT)   # replace the mock heuristic extractor
    e.ingest(NoteInput(text=TURING, created_at="2026-07-01T10:00:00Z"))
    e.ingest(NoteInput(text=PAPER, created_at="2026-07-02T10:00:00Z"))
    yield e
    e.close()


def _entity(eng, name):
    return next(n for n in eng._g.store.nodes_of_type(NodeType.ENTITY)
                if n.name == name)


def _node(gp, nid):
    return next(n for n in gp["nodes"] if n["id"] == nid)


def test_episode_root_carries_fact_edges_and_stub_counts(eng):
    ep = eng.episodes_list()["episodes"][0]["id"]      # the TURING episode
    gp = eng.graph_preview(ep)
    root = _node(gp, ep)
    assert root["kind"] == "episode" and root["hop"] == 0
    assert root["category"] is None
    assert TURING.startswith(root["name"][:20])        # text label, not source_ref

    alan = _entity(eng, "Alan Turing")
    bletchley = _entity(eng, "Bletchley Park")
    enigma = _entity(eng, "Enigma")
    assert {n["id"] for n in gp["nodes"]} == {ep, alan.id, bletchley.id, enigma.id}
    assert all(n["hop"] == 1 for n in gp["nodes"] if n["id"] != ep)
    assert _node(gp, enigma.id)["kind"] == "concept"
    assert _node(gp, alan.id)["kind"] == "entity"
    assert _node(gp, alan.id)["category"] == "person"

    mentions = [e for e in gp["edges"] if e["etype"] == "MENTIONS"]
    assert {(e["src"], e["dst"]) for e in mentions} == \
        {(ep, alan.id), (ep, bletchley.id), (ep, enigma.id)}
    assert all(e["label"] == "" for e in mentions)
    facts = [e for e in gp["edges"] if e["etype"] == "RELATED_TO"]
    assert facts == [{"src": alan.id, "dst": bletchley.id,
                      "etype": "RELATED_TO", "label": "worked_at"}]

    # Alan and Enigma also appear in the PAPER episode, which is off-screen here.
    assert _node(gp, alan.id)["external_connections"] == 1
    assert _node(gp, enigma.id)["external_connections"] == 1
    assert _node(gp, bletchley.id)["external_connections"] == 0
    assert root["external_connections"] == 0


def test_entity_root_returns_episodes_and_fact_partners(eng):
    alan = _entity(eng, "Alan Turing")
    bletchley = _entity(eng, "Bletchley Park")
    gp = eng.graph_preview(alan.id)
    root = _node(gp, alan.id)
    assert root["hop"] == 0 and root["kind"] == "entity"

    eps = [e["id"] for e in eng.episodes_list()["episodes"]]
    # neighbourhood = both mentioning episodes + the worked_at fact partner
    assert {n["id"] for n in gp["nodes"]} == {alan.id, bletchley.id, *eps}
    ep_nodes = [n for n in gp["nodes"] if n["kind"] == "episode"]
    assert all(n["hop"] == 1 for n in ep_nodes)
    assert all(n["name"] and n["name"] not in ("app", "capture") for n in ep_nodes)

    # the complete one-hop graph: every MENTIONS edge between two DRAWN nodes rides
    # along, so the TURING episode also links to Bletchley Park, not just to the root
    turing_ep = next(e["id"] for e in eng.episodes_list()["episodes"]
                     if "worked at" in (eng.episode(e["id"]) or {}).get("text", ""))
    assert {(e["src"], e["dst"]) for e in gp["edges"] if e["etype"] == "MENTIONS"} == \
        {(ep, alan.id) for ep in eps} | {(turing_ep, bletchley.id)}
    assert {(e["src"], e["dst"], e["label"]) for e in gp["edges"]
            if e["etype"] == "RELATED_TO"} == {(alan.id, bletchley.id, "worked_at")}
    assert root["external_connections"] == 0           # everything fits on screen


def test_concept_root_and_not_found(eng):
    enigma = _entity(eng, "Enigma")
    gp = eng.graph_preview(enigma.id)
    assert _node(gp, enigma.id)["kind"] == "concept"
    assert _node(gp, enigma.id)["hop"] == 0
    assert len([n for n in gp["nodes"] if n["kind"] == "episode"]) == 2
    with pytest.raises(NotFound):
        eng.graph_preview("nope_123")
    with pytest.raises(NotFound):                      # a tag id is not a graph root
        tag = eng._g.store.nodes_of_type(NodeType.TAG)[0]
        eng.graph_preview(tag.id)


def test_wire_shape_carries_predicate_and_stubs(eng):
    alan = _entity(eng, "Alan Turing")
    wire = Daemon._wire_graph_preview(eng.graph_preview(alan.id), alan.id)
    root = next(n for n in wire["nodes"] if n["id"] == alan.id)
    assert root == {"id": alan.id, "label": "Alan Turing", "kind": "entity",
                    "hop": 0, "external_connections": 0, "entity_category": "person"}
    fact = next(e for e in wire["edges"] if e["kind"] == "RELATED_TO")
    assert fact["label"] == "worked_at"                # predicate on the wire label
    assert all(e["label"] == "" for e in wire["edges"] if e["kind"] == "MENTIONS")
    # Alan's one-hop neighbourhood has no concept (Enigma is only co-mentioned, not a
    # fact partner); the episode-rooted preview carries all three node kinds.
    assert {n["kind"] for n in wire["nodes"]} == {"entity", "episode"}
    ep = eng.episodes_list()["episodes"][0]["id"]
    wire_ep = Daemon._wire_graph_preview(eng.graph_preview(ep), ep)
    assert {n["kind"] for n in wire_ep["nodes"]} <= {"entity", "episode", "concept"}
    assert "concept" in {n["kind"] for n in wire_ep["nodes"]}
