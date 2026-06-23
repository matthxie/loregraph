"""Tests for the agentic graph-traversal query (kg/agent.py).

Fully offline/deterministic like test_kg.py: hashing embedder + heuristic extractor, plus
a scripted fake Anthropic client (mirroring _FakeL3Client) that exercises the real
tool-use loop with no API key or network. Run: python -m pytest -q
"""
from __future__ import annotations

import types

import pytest

from kg import Config, KnowledgeGraph
from kg.agent import (AgentAnswer, ClaudeAgent, GraphTools, OfflineAgent, _validate_citations,
                      get_agent)
from kg.canonicalize import Canonicalizer
from kg.corpus import CorpusItem
from kg.embedders import get_embedder
from kg.models import Edge, EdgeType, EntityType, Modality, Provenance, entity_node, \
    object_node, relation_tag_node
from kg.store import GraphStore


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def cfg() -> Config:
    c = Config.default()
    c.embedder = "hashing"
    c.extractor = "heuristic"
    return c


def sample_items():
    return [
        CorpusItem(id="a", modality="text", source_ref="u/a", title="Alan Turing",
                   text="Alan Turing was a British mathematician and computer scientist. "
                        "Turing worked at Bletchley Park on cryptography during the war. "
                        "He is considered a father of computer science."),
        CorpusItem(id="b", modality="text", source_ref="u/b", title="Bletchley Park",
                   text="Bletchley Park was the central British codebreaking site during "
                        "World War II. Alan Turing and mathematicians worked there on "
                        "cryptography and the Enigma machine."),
        CorpusItem(id="c", modality="text", source_ref="u/c", title="Photosynthesis",
                   text="Photosynthesis is the process by which plants convert light energy "
                        "into chemical energy. Chlorophyll absorbs sunlight in the leaves."),
        CorpusItem(id="d", modality="image", source_ref="img/d.jpg",
                   image_path="img/d.jpg", label_hint="dog, frisbee, person"),
    ]


def agent_graph(build_comms: bool = True) -> KnowledgeGraph:
    import os
    import tempfile
    g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), cfg())
    g.ingest(sample_items())
    if build_comms:
        g.build_communities()
    return g


def tools_of(g: KnowledgeGraph) -> GraphTools:
    return GraphTools(g.store, g.embedder, g.canon, g.config)


# ---- scripted fake Anthropic client (mirrors _FakeL3Client) ---------------- #
def _tool_use(tid, name, inp):
    return types.SimpleNamespace(type="tool_use", id=tid, name=name, input=inp)


def _text(t):
    return types.SimpleNamespace(type="text", text=t)


def _turn(*blocks):
    return types.SimpleNamespace(content=list(blocks), stop_reason="tool_use")


class _FakeAgentClient:
    """Emits a scripted sequence of assistant turns; records every create() it was
    offered so tests can assert the tool set + message sizes."""
    def __init__(self, script):
        self._script = list(script)
        self.messages = self
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self._script.pop(0)


# --------------------------------------------------------------------------- #
# per-tool executor tests (no LLM)
# --------------------------------------------------------------------------- #
def test_tool_seed_and_spread_compact_stubs():
    t = tools_of(agent_graph())
    out = t.dispatch("seed_and_spread", {"query": "cryptography Bletchley codebreaking", "k": 5})
    assert out["objects"], "expected ranked objects"
    o = out["objects"][0]
    assert {"id", "name", "ntype", "score", "snippet"} <= set(o)
    assert "raw_text" not in o and len(o["snippet"]) <= 160
    ids = {o["id"] for o in out["objects"]}
    assert {"obj_a", "obj_b"} & ids
    assert all(o["ntype"] == "object" for o in out["objects"])


def test_tool_keyword_search_matches_bm25():
    g = agent_graph()
    t = tools_of(g)
    out = t.dispatch("keyword_search", {"query": "Enigma cryptography", "k": 6})
    got = [o["id"] for o in out["objects"]]
    want = [oid for oid, _ in t.seeder.bm25_search("Enigma cryptography", k=6)][:len(got)]
    assert got == want and got, "keyword_search must agree with Seeder.bm25_search"


def test_tool_vector_search_compact():
    t = tools_of(agent_graph())
    out = t.dispatch("vector_search", {"query": "plants light energy", "k": 4})
    assert out["objects"] and all("score" in o and o["ntype"] == "object"
                                  for o in out["objects"])


def test_tool_neighbors_label_direction_provenance_parallel():
    """CONSTRAINT 4: neighbors surfaces label + direction + provenance + confidence, and
    parallel relationship tags between a pair yield one row each (rev 4)."""
    store = GraphStore(cfg())
    store.add_node(entity_node("entity_a", name="Alice", etype=EntityType.PERSON, ts="t"))
    store.add_node(entity_node("entity_b", name="Bob", etype=EntityType.PERSON, ts="t"))
    store.add_node(relation_tag_node("rel_0000", canonical="works_with", ts="t"))
    store.add_node(relation_tag_node("rel_0001", canonical="is_friend_of", ts="t"))
    store.add_edge(Edge("entity_a", "entity_b", EdgeType.RELATED_TO, Provenance.EXTRACTED,
                        0.9, 0.9, rel_tag="rel_0000"))
    store.add_edge(Edge("entity_a", "entity_b", EdgeType.RELATED_TO, Provenance.EXTRACTED,
                        0.8, 0.8, rel_tag="rel_0001"))
    t = GraphTools(store, get_embedder(cfg()), Canonicalizer(store, get_embedder(cfg()), cfg()),
                   cfg())
    out = t.dispatch("neighbors", {"node_id": "entity_a", "direction": "out",
                                   "etypes": ["RELATED_TO"]})
    assert {e["rel"] for e in out["edges"]} == {"works_with", "is_friend_of"}  # parallel, labeled
    assert all(e["direction"] == "out" and e["provenance"] == "EXTRACTED"
               and "confidence" in e for e in out["edges"])
    # the reverse node sees them as incoming
    back = t.dispatch("neighbors", {"node_id": "entity_b", "direction": "in"})
    assert {e["direction"] for e in back["edges"]} == {"in"}
    assert back["edges"] and all(e["neighbor"]["id"] == "entity_a" for e in back["edges"])


def test_tool_neighbors_direction_matches_store():
    g = agent_graph()
    t = tools_of(g)
    out = t.dispatch("neighbors", {"node_id": "obj_a", "direction": "out", "limit": 20})
    store_out = {nbr for nbr, _ in g.store.neighbors("obj_a", direction="out")}
    assert all(e["neighbor"]["id"] in store_out for e in out["edges"])


def test_tool_find_path_labeled_bounded_and_disconnected():
    store = GraphStore(cfg())
    store.add_node(entity_node("entity_a", name="Alice", etype=EntityType.PERSON, ts="t"))
    store.add_node(entity_node("entity_b", name="Bob", etype=EntityType.PERSON, ts="t"))
    store.add_node(entity_node("entity_c", name="Carol", etype=EntityType.PERSON, ts="t"))  # isolated
    store.add_node(relation_tag_node("rel_0000", canonical="works_with", ts="t"))
    store.add_edge(Edge("entity_a", "entity_b", EdgeType.RELATED_TO, Provenance.EXTRACTED,
                        0.9, 0.9, rel_tag="rel_0000"))
    t = GraphTools(store, get_embedder(cfg()), Canonicalizer(store, get_embedder(cfg()), cfg()),
                   cfg())
    out = t.dispatch("find_path", {"source_id": "entity_a", "target_id": "entity_b"})
    assert out["found"] and out["hops"] == 1
    assert out["path"][1]["rel_in"] == "works_with" and out["path"][1]["direction"] == "out"
    # disconnected
    assert t.dispatch("find_path", {"source_id": "entity_a", "target_id": "entity_c"})["found"] is False
    # bounded: a 1-hop path is rejected when max_hops would require fewer than available
    far = t.dispatch("find_path", {"source_id": "entity_a", "target_id": "entity_b",
                                   "max_hops": 1})
    assert far["found"] is True  # exactly 1 hop is allowed


def test_tool_read_object_full_text_and_rejects_non_object():
    g = agent_graph()
    t = tools_of(g)
    out = t.dispatch("read_object", {"object_id": "obj_a"})
    assert out["text"] and "Turing" in out["text"] and len(out["text"]) <= g.config.agent_read_chars
    assert out["modality"] == "text" and isinstance(out["tags"], list)
    # image object returns its description as text
    img = t.dispatch("read_object", {"object_id": "obj_d"})
    assert img["modality"] == "image" and img["text"]
    # non-object id is rejected, not read
    ent = next(n.id for n in g.store.nodes.values() if n.ntype.value == "entity")
    assert "error" in t.dispatch("read_object", {"object_id": ent})
    assert "error" in t.dispatch("read_object", {"object_id": "nope"})


def test_tool_browse_themes_before_and_after_communities():
    g = agent_graph(build_comms=False)
    t = tools_of(g)
    pre = t.dispatch("browse_themes", {"query": "themes"})
    assert pre["themes"] == [] and "note" in pre
    g.build_communities()
    t2 = tools_of(g)
    # query with the community's own summary so the match is robust to the hashing
    # embedder's floor (a real query↔summary cosine is solidly positive under bge)
    from kg.models import NodeType
    summary = g.store.nodes_of_type(NodeType.COMMUNITY)[0].summary
    post = t2.dispatch("browse_themes", {"query": summary})
    assert post["themes"] and "members" in post["themes"][0]
    assert len(post["themes"][0]["summary"]) <= 200


def test_node_budget_truncates():
    g = agent_graph()
    c = g.config
    c.agent_node_budget = 2
    t = GraphTools(g.store, g.embedder, g.canon, c)
    out = t.dispatch("neighbors", {"node_id": "obj_a", "direction": "both", "limit": 20})
    assert out["truncated"] is True
    assert len(out["edges"]) <= 2 and len(t.touched) <= 2


def test_seed_and_spread_respects_budget():
    """Regression: seed_and_spread's seed loop must obey the node budget and flag it (the
    seed path used to bypass the budget and report ceiling_hit=False)."""
    g = agent_graph()
    c = g.config
    c.agent_node_budget = 3
    t = GraphTools(g.store, g.embedder, g.canon, c)
    out = t.dispatch("seed_and_spread", {"query": "cryptography Bletchley Turing Enigma", "k": 8})
    assert len(t.touched) <= 3
    assert out["ceiling_hit"] is True and out["truncated"] is True


def test_browse_themes_members_respect_budget():
    """Regression: emitted theme members must equal the touched (budgeted) ids."""
    from kg.models import NodeType
    g = agent_graph()
    c = g.config
    c.agent_node_budget = 1
    t = GraphTools(g.store, g.embedder, g.canon, c)
    summary = g.store.nodes_of_type(NodeType.COMMUNITY)[0].summary
    out = t.dispatch("browse_themes", {"query": summary})
    surfaced = [m for th in out["themes"] for m in th["members"]]
    assert len(surfaced) <= 1 and set(surfaced) <= t.touched


# --------------------------------------------------------------------------- #
# real loop via injected fake client
# --------------------------------------------------------------------------- #
def _submit(answer, citations):
    return _turn(_tool_use("u_sub", "submit_answer",
                           {"answer": answer, "citations": citations}))


def test_agent_loop_traverses_and_cites():
    g = agent_graph()
    script = [
        _turn(_tool_use("u1", "seed_and_spread", {"query": "Turing cryptography"})),
        _turn(_tool_use("u2", "read_object", {"object_id": "obj_a"})),
        _submit("Turing worked on cryptography at Bletchley Park.", ["obj_a"]),
    ]
    client = _FakeAgentClient(script)
    ans = g.ask("What did Turing do?", client=client)
    assert ans.backend == "claude" and ans.steps == 3 and ans.stopped == "answered"
    assert [s["tool"] for s in ans.trace] == ["seed_and_spread", "read_object"]
    assert ans.citations == ["obj_a"] and ans.dropped_citations == []
    assert "obj_a" in ans.touched and "obj_a" in ans.object_ids
    # every create() was offered the read tools + the submit terminal
    offered = {t["name"] for t in client.calls[0]["tools"]}
    assert "seed_and_spread" in offered and "submit_answer" in offered


def test_agent_drops_hallucinated_citation():
    g = agent_graph()
    script = [
        _turn(_tool_use("u1", "seed_and_spread", {"query": "Turing"})),
        _turn(_tool_use("u2", "read_object", {"object_id": "obj_a"})),
        _submit("see obj_a and obj_zzz", ["obj_a", "obj_zzz", "entity_0000"]),
    ]
    ans = g.ask("q", client=_FakeAgentClient(script))
    assert ans.citations == ["obj_a"]
    assert "obj_zzz" in ans.dropped_citations and "entity_0000" in ans.dropped_citations
    assert ans.notes and "dropped" in ans.notes[0]


def test_agent_cite_without_reading_is_dropped():
    """A real object that was surfaced by search but never read must NOT be citable."""
    g = agent_graph()
    script = [
        _turn(_tool_use("u1", "seed_and_spread", {"query": "Bletchley"})),
        _submit("answer", ["obj_b"]),   # obj_b surfaced but never read_object'd
    ]
    ans = g.ask("q", client=_FakeAgentClient(script))
    assert ans.citations == [] and "obj_b" in ans.dropped_citations


def test_agent_step_cap_forces_submit():
    g = agent_graph()
    # never emits submit during the loop; the forced final turn does
    script = [
        _turn(_tool_use("u1", "read_object", {"object_id": "obj_a"})),
        _turn(_tool_use("u2", "read_object", {"object_id": "obj_a"})),
        _submit("forced answer from evidence", ["obj_a"]),   # the _force_submit turn
    ]
    ans = g.ask("q", client=_FakeAgentClient(script), max_steps=2)
    assert ans.stopped == "step_cap" and ans.answer and ans.citations == ["obj_a"]


def test_agent_prose_without_submit_is_salvaged():
    g = agent_graph()
    script = [
        _turn(_tool_use("u1", "read_object", {"object_id": "obj_a"})),
        _turn(_text("Turing worked at Bletchley Park (obj_a).")),  # prose, no tool
    ]
    ans = g.ask("q", client=_FakeAgentClient(script))
    assert ans.stopped == "prose" and "Bletchley" in ans.answer
    assert ans.citations == ["obj_a"]   # obj_a was read, so the salvaged id validates


def test_agent_repeat_call_deduped():
    g = agent_graph()
    script = [
        _turn(_tool_use("u1", "seed_and_spread", {"query": "Turing"})),
        _turn(_tool_use("u2", "seed_and_spread", {"query": "Turing"})),  # identical → deduped
        _submit("done", []),
    ]
    ans = g.ask("q", client=_FakeAgentClient(script))
    assert ans.trace[0]["result_summary"] != "(deduped)"
    assert ans.trace[1]["result_summary"] == "(deduped)"


def test_agent_tool_result_size_capped():
    g = agent_graph()
    g.config.agent_result_chars = 40
    t = GraphTools(g.store, g.embedder, g.canon, g.config)
    client = _FakeAgentClient([
        _turn(_tool_use("u1", "seed_and_spread", {"query": "cryptography Bletchley"})),
        _submit("done", []),
    ])
    ClaudeAgent(t, g.config, client=client).run("q")
    # the second create() carries the tool_result we fed back — it must be truncated
    tool_results = [b for m in client.calls[1]["messages"] if m["role"] == "user"
                    and isinstance(m["content"], list) for b in m["content"]
                    if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert tool_results and all(len(b["content"]) <= 40 for b in tool_results)


# --------------------------------------------------------------------------- #
# offline agent
# --------------------------------------------------------------------------- #
def test_offline_agent_answers_and_cites():
    g = agent_graph()
    ans = g.ask("How are Alan Turing and Bletchley Park connected?", backend="offline")
    assert ans.backend == "offline" and ans.answer
    assert ans.citations and ans.dropped_citations == []
    assert all(g.store.get_node(c).ntype.value == "object" for c in ans.citations)
    tools = [s["tool"] for s in ans.trace]
    assert "seed_and_spread" in tools and "read_object" in tools
    # the connection cue drove a find_path call (entities exist in the sample)
    assert "find_path" in tools


def test_offline_agent_global_routes_to_themes():
    g = agent_graph()
    ans = g.ask("what are the main themes across the collection", backend="offline")
    assert ans.trace and ans.trace[0]["tool"] == "browse_themes"


def test_get_agent_offline_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    g = agent_graph()
    c = Config.default()
    c.embedder, c.extractor = "hashing", "heuristic"   # agent_backend stays "auto"
    a = get_agent(g.store, g.embedder, g.canon, c)
    assert isinstance(a, OfflineAgent)


def test_offline_and_online_trace_same_shape():
    g = agent_graph()
    off = g.ask("cryptography at Bletchley", backend="offline")
    on = g.ask("cryptography at Bletchley", client=_FakeAgentClient([
        _turn(_tool_use("u1", "seed_and_spread", {"query": "cryptography Bletchley"})),
        _turn(_tool_use("u2", "read_object", {"object_id": "obj_b"})),
        _submit("ok", ["obj_b"]),
    ]))
    keys = {"step", "tool", "input", "result_summary"}
    for ans in (off, on):
        assert isinstance(ans, AgentAnswer)
        assert all(set(s) == keys for s in ans.trace)
        assert isinstance(ans.object_ids, list)


# --------------------------------------------------------------------------- #
# eval interop
# --------------------------------------------------------------------------- #
def test_agent_object_ids_feed_recall():
    from kg.evaluate import _recall_at_k, cross_article_questions, evaluate
    g = agent_graph()
    ans = g.ask("Alan Turing", backend="offline")
    assert 0.0 <= _recall_at_k(ans.object_ids, {"obj_a"}, 8) <= 1.0
    # the eval harness accepts "agent" as a mode and runs fully offline
    qs = cross_article_questions(g, limit=2)
    if qs:
        scores = evaluate(g, qs, modes=("agent",), k=5)
        assert len(scores) == 1 and 0.0 <= scores[0].recall_at_k <= 1.0


# --------------------------------------------------------------------------- #
# viewer adapter + CLI
# --------------------------------------------------------------------------- #
def test_agent_trace_payload_matches_viewer_schema():
    from kg.viz import agent_trace_payload, graph_payload, render_html
    g = agent_graph()
    ans = g.ask("cryptography Bletchley Turing", backend="offline")
    tr = agent_trace_payload(ans, g.store)
    assert tr["mode"] == "agent"
    assert set(tr) >= {"query", "mode", "nodes", "edges", "ranked", "seeds", "hops"}
    assert "note" not in tr   # a note would short-circuit the viewer to an empty render
    assert all(0.0 <= n["x"] <= 1.0 and 0.0 <= n["y"] <= 1.0 for n in tr["nodes"])
    html = render_html(graph_payload(g.store), trace=tr, server=False)
    assert "/*__DATA__*/" not in html and "<svg" in html


def test_cmd_ask_offline_prints_answer_and_citations(capsys):
    import os
    import tempfile
    from kg.cli import build_parser
    store = os.path.join(tempfile.mkdtemp(), "kg.db")
    g = KnowledgeGraph.open(store, cfg())
    g.ingest(sample_items())
    g.save()
    args = build_parser().parse_args(["--store", store, "ask",
                                      "How is Turing connected to Bletchley Park?",
                                      "--backend", "offline", "--show-trace"])
    args.func(args)
    out = capsys.readouterr().out
    assert "backend=offline" in out and "answer:" in out and "[obj_" in out
    assert "trace:" in out


def test_validate_citations_unit():
    g = agent_graph()
    from kg.agent import Trace
    tr = Trace()
    tr.read.update({"obj_a"})
    kept, dropped = _validate_citations(["obj_a", "obj_a", "obj_b", "entity_0000", "nope"],
                                        tr, g.store)
    assert kept == ["obj_a"]                          # only the read object survives, deduped
    assert set(dropped) == {"obj_b", "entity_0000", "nope"}
