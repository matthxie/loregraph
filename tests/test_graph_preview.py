"""graph_preview contract tests (PROTOCOL §3.6a): entity/concept roots, predicate
labels on fact edges, and external_connections for off-screen continuation stubs.

Fully offline: ScriptedExtractor feeds known Extractions (no LLM); embeddings use the
local bge model, same policy as the rest of the suite.
"""
from __future__ import annotations

import tempfile

import pytest

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
    ep = eng.episodes_list()["episodes"][-1]["id"]     # the TURING episode (oldest;
    #                                                    §7.2 rows are newest-first)
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


def test_episode_detail_splits_concepts_from_entities(eng):
    """episode() reports CONCEPT-type nodes in `concepts` (topical strings), never folded
    into `entities`/`entity_categories` — so clients can count them as their own category."""
    eps = eng.episodes_list()["episodes"]              # newest-first (§7.2)
    turing = eng.episode(eps[-1]["id"])                # "Alan Turing … Bletchley Park … Enigma"
    assert set(turing["entities"]) == {"Alan Turing", "Bletchley Park"}
    assert turing["concepts"] == ["Enigma"]
    assert "Enigma" not in turing["entities"]
    assert "Enigma" not in turing["entity_categories"]
    assert turing["entity_categories"]["Alan Turing"] == "person"
    assert turing["entity_categories"]["Bletchley Park"] == "place"

    paper = eng.episode(eps[0]["id"])                  # "Turing wrote a paper about the Enigma"
    assert set(paper["entities"]) == {"Alan Turing"}
    assert paper["concepts"] == ["Enigma"]


def test_episode_detail_serves_grounded_facts(eng):
    """episode() carries the §3.6 facts this note grounds — the fact rows whose
    provenance episode_id is this note, with the structured §3.5 field shape."""
    eps = eng.episodes_list()["episodes"]
    turing_id = eps[-1]["id"]                          # the fact-bearing TURING note
    facts = eng.episode(turing_id)["facts"]
    assert len(facts) == 1
    f = facts[0]
    assert (f["source"], f["predicate"], f["target"]) == \
        ("Alan Turing", "worked_at", "Bletchley Park")
    assert f["status"] == "asserted" and f["episode_id"] == turing_id
    assert "worked_at" in f["rendered"]
    assert eng.episode(eps[0]["id"])["facts"] == []    # the PAPER note grounds none


# --- episode_graph: the provenance subgraph for one ingest (§3.6a) ------------------
# Two notes relate the SAME pair differently, so "one hop from the episode" and "what
# this ingest created" diverge — the earlier note's fact is on screen for a one-hop
# preview of the later note (both entities are drawn) but is NOT part of the later note.
COLLEAGUES = "Sam and Alex are colleagues."
LUNCH = "Had lunch with Sam and Alex."

PROV_SCRIPT = {
    COLLEAGUES: Extraction(
        entities=[ExtractedEntity("Sam", EntityType.PERSON),
                  ExtractedEntity("Alex", EntityType.PERSON)],
        relations=[ExtractedRelation(source="Sam", target="Alex",
                                     labels=["colleague_of"],
                                     provenance=Provenance.EXTRACTED, confidence=0.9)],
    ),
    LUNCH: Extraction(
        entities=[ExtractedEntity("Sam", EntityType.PERSON),
                  ExtractedEntity("Alex", EntityType.PERSON)],
        relations=[ExtractedRelation(source="Sam", target="Alex",
                                     labels=["had_lunch_with"],
                                     provenance=Provenance.EXTRACTED, confidence=0.9)],
    ),
}


@pytest.fixture
def prov_eng():
    e = Engine.open(tempfile.mkdtemp(), {"kind": "mock"})
    e._g.extractor = ScriptedExtractor(PROV_SCRIPT)
    e.ingest(NoteInput(text=COLLEAGUES, created_at="2026-07-01T10:00:00Z"))
    e.ingest(NoteInput(text=LUNCH, created_at="2026-07-02T10:00:00Z"))
    yield e
    e.close()


def _lunch_id(eng):
    return eng.episodes_list()["episodes"][0]["id"]      # newest-first (§7.2)


def test_episode_graph_scopes_edges_to_this_ingest(prov_eng):
    """episode_graph shows only the facts THIS note asserted; the colleague_of relation
    (from the earlier note) is excluded from the lunch note's graph even though both of
    its endpoints are on screen — where the one-hop graph_preview carries it."""
    lunch = _lunch_id(prov_eng)
    sam = _entity(prov_eng, "Sam")
    alex = _entity(prov_eng, "Alex")

    g = prov_eng.episode_graph(lunch)
    assert {n["id"] for n in g["nodes"]} == {lunch, sam.id, alex.id}
    assert {(e["src"], e["dst"], e["label"]) for e in g["edges"]
            if e["etype"] == "RELATED_TO"} == {(sam.id, alex.id, "had_lunch_with")}

    # The one-hop preview is NOT provenance-scoped: it carries both facts between the pair.
    assert {(e["src"], e["dst"], e["label"])
            for e in prov_eng.graph_preview(lunch)["edges"]
            if e["etype"] == "RELATED_TO"} == {(sam.id, alex.id, "had_lunch_with"),
                                               (sam.id, alex.id, "colleague_of")}


def test_episode_graph_keeps_mentions_and_external_stubs(prov_eng):
    """Every mentioned entity keeps its episode→entity MENTIONS spoke, and an entity that
    also lives in a memory outside this ingest reports external_connections, so the client
    still draws the dashed continuation stub to the rest of the graph."""
    lunch = _lunch_id(prov_eng)
    sam = _entity(prov_eng, "Sam")
    alex = _entity(prov_eng, "Alex")
    g = prov_eng.episode_graph(lunch)

    assert {(e["src"], e["dst"]) for e in g["edges"] if e["etype"] == "MENTIONS"} == \
        {(lunch, sam.id), (lunch, alex.id)}
    assert all(e["label"] == "" for e in g["edges"] if e["etype"] == "MENTIONS")

    assert _node(g, lunch)["hop"] == 0
    assert all(n["hop"] == 1 for n in g["nodes"] if n["id"] != lunch)
    # Both are also in the colleagues note (off-screen here) → one stub apiece.
    assert _node(g, sam.id)["external_connections"] == 1
    assert _node(g, alex.id)["external_connections"] == 1


def test_episodes_list_embeds_raw_graph(prov_eng):
    """The list row's embedded graph is the RAW store subgraph (the dev view), so the memory
    card and its detail agree: mention nodes are present and only this note's fact rides."""
    row = prov_eng.episodes_list()["episodes"][0]        # the lunch note (newest)
    kinds = {n["kind"] for n in row["graph_preview"]["nodes"]}
    assert "mention" in kinds and "entity" in kinds and "episode" in kinds
    labels = {e["label"] for e in row["graph_preview"]["edges"]
              if e["etype"] == "RELATED_TO"}
    assert labels == {"had_lunch_with"}                  # NOT colleague_of (other note)


# --- episode_raw_graph: the raw store structure for one ingest (dev view) ------------
def test_episode_raw_graph_exposes_store_structure(eng):
    """The raw graph draws the engine's real nodes and edges for a note — the episode, a
    MENTION per occurrence, the ENTITY anchors they RESOLVES_TO, the TAGGED_AS topic tag,
    and the entity→entity fact — uncollapsed, with the raw edge types as labels."""
    ep = eng.episodes_list()["episodes"][-1]["id"]       # the TURING note (has a tag + fact)
    g = eng.episode_raw_graph(ep)

    kinds = [n["kind"] for n in g["nodes"]]
    assert kinds.count("episode") == 1
    assert kinds.count("mention") == 3                   # Alan, Bletchley, Enigma occurrences
    assert kinds.count("entity") == 3
    assert kinds.count("tag") == 1                       # "cryptography"
    assert _node(g, ep)["hop"] == 0
    assert all(n["hop"] == 1 for n in g["nodes"] if n["id"] != ep)
    # A concept entity keeps its raw NodeType (entity), carrying its type in `category`.
    enigma = _entity(eng, "Enigma")
    assert _node(g, enigma.id) == {**_node(g, enigma.id), "kind": "entity", "category": "concept"}

    etypes = [e["etype"] for e in g["edges"]]
    assert etypes.count("MENTIONED_IN") == 3
    assert etypes.count("RESOLVES_TO") == 3
    assert etypes.count("TAGGED_AS") == 1
    fact = next(e for e in g["edges"] if e["etype"] == "RELATED_TO")
    assert fact["label"] == "worked_at"                  # the predicate rides as the label
    # The fact is entity→entity (not mention→ or episode→).
    assert {_node(g, fact["src"])["kind"], _node(g, fact["dst"])["kind"]} == {"entity"}


def test_episode_raw_graph_is_provenance_scoped(prov_eng):
    """Even raw, the graph is scoped to this note: a fact another note asserted between the
    same two entities (colleague_of) is not drawn, and a shared entity reports the neighbours
    it has outside this ingest via external_connections (the dashed stub)."""
    lunch = _lunch_id(prov_eng)
    g = prov_eng.episode_raw_graph(lunch)
    assert {e["label"] for e in g["edges"] if e["etype"] == "RELATED_TO"} == {"had_lunch_with"}
    sam = _entity(prov_eng, "Sam")
    # Sam's mention/entity also live in the colleagues note → the entity has off-ingest links.
    assert _node(g, sam.id)["external_connections"] >= 1


# --- node_raw_graph: click a node to re-root the raw graph on it ---------------------
def test_node_raw_graph_reroots_on_an_entity(eng):
    """Clicking an entity re-roots the raw graph on it (hop 0) and draws what connects to it —
    its mention occurrences and its fact partner — with the real edge types."""
    alan = _entity(eng, "Alan Turing")
    bletchley = _entity(eng, "Bletchley Park")
    g = eng.node_raw_graph(alan.id)
    assert _node(g, alan.id)["hop"] == 0 and _node(g, alan.id)["kind"] == "entity"
    assert "mention" in {n["kind"] for n in g["nodes"] if n["id"] != alan.id}
    assert any(e["etype"] == "RESOLVES_TO" for e in g["edges"])       # mention → Alan
    fact = next(e for e in g["edges"] if e["etype"] == "RELATED_TO")
    assert fact["label"] == "worked_at" and bletchley.id in (fact["src"], fact["dst"])


def test_node_raw_graph_on_a_tag_lists_its_episodes(eng):
    """Re-rooting on a tag shows every episode TAGGED_AS it — the two cryptography notes."""
    tag = next(n for n in eng._g.store.nodes_of_type(NodeType.TAG) if n.name == "cryptography")
    g = eng.node_raw_graph(tag.id)
    assert _node(g, tag.id)["hop"] == 0 and _node(g, tag.id)["kind"] == "tag"
    assert len([n for n in g["nodes"] if n["kind"] == "episode"]) == 2
    tagged = {(e["src"], e["dst"]) for e in g["edges"] if e["etype"] == "TAGGED_AS"}
    assert len(tagged) == 2 and all(dst == tag.id for _src, dst in tagged)


def test_node_raw_graph_on_a_mention_shows_its_star(eng):
    """A mention connects to exactly its episode (MENTIONED_IN) and entity (RESOLVES_TO)."""
    m = next(n for n in eng._g.store.nodes_of_type(NodeType.MENTION)
             if n.name == "Bletchley Park")
    g = eng.node_raw_graph(m.id)
    assert _node(g, m.id)["kind"] == "mention" and _node(g, m.id)["hop"] == 0
    etypes = {e["etype"] for e in g["edges"]}
    assert "MENTIONED_IN" in etypes and "RESOLVES_TO" in etypes
    assert {"episode", "entity"} <= {n["kind"] for n in g["nodes"] if n["id"] != m.id}


def test_node_raw_graph_unknown_id(eng):
    with pytest.raises(NotFound):
        eng.node_raw_graph("nope_123")
