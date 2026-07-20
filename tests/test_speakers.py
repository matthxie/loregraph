"""Round 8 — speaker attribution (config.speaker_attribution): distinguish user-stated facts
from assistant-suggested/quoted material so the reader refuses to compute an answer from
figures it can't attribute to the user (docs/OFFLINE_EVAL.md Round 8, the *_abs failures).

Contracts under test:
  * parse: "User:"/"Assistant:" turn markers → speaker_id + kind; both roles → 'mixed'; no
    marker → (None, None); stable/distinct ids;
  * registry: upsert is idempotent; persists + reloads (incl. a pre-feature db with no table);
  * attribution: DERIVED any-user reduction over episode_id ∪ confirmed_by — an assistant-only
    fact is assistant-grounded, a user-echoed one is user-grounded, mixed counts as human,
    unknown provenance stays unmarked;
  * marker: the FACTS line resting only on assistant turns gets "[assistant]", a user-echoed
    line does NOT — only when the knob is on;
  * backfill: idempotent + incremental on a scripted store AND on a COPY of a real cached
    store; does not change the ingest-cache key;
  * knob OFF ⇒ context AND system prompt byte-identical.

Offline/deterministic: no LLM (ScriptedExtractor / a fake answer client), local regex parse.
"""
from __future__ import annotations

import json
import os
import tempfile
import types
from dataclasses import replace

import pytest

import kg.graph as kg_graph
from kg import Config, KnowledgeGraph
from kg.canonicalize import Canonicalizer
from kg.corpus import CorpusItem
from kg.embedders import get_embedder
from kg.extractors import (ExtractedEntity, ExtractedRelation, Extraction,
                           ScriptedExtractor)
from kg.ingest_cache import INGEST_RELEVANT_FIELDS, ingest_cache_key
from kg.models import (EntityType, Modality, NodeType, Provenance, entity_node,
                       episode_node)
from kg.rag import _RAG_SYS, _SPEAKER_RULE, ContextBuilder, OpenAIAnswerer
from kg.retrieval import RetrievalResult
from kg.speakers import (asserted_by, assistant_marker, backfill_speakers, detect_roles,
                         ensure_speaker, is_assistant_only, parse_speaker, speaker_row_for,
                         stamp_episode)
from kg.store import GraphStore
from kg.temporal import apply_fact


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(kg_graph, "get_extractor", lambda config: ScriptedExtractor({}))


# --------------------------------------------------------------------------- #
# 1. parse
# --------------------------------------------------------------------------- #
def test_parse_user():
    sid, kind = parse_speaker("User: I spent $80 fixing the sink last week.")
    assert kind == "human" and sid == speaker_row_for("human").speaker_id


def test_parse_assistant():
    sid, kind = parse_speaker("Assistant: Typical airport-bus fares run 2000–3000 JPY.")
    assert kind == "assistant" and sid == speaker_row_for("assistant").speaker_id


def test_parse_mixed_both_roles_present():
    """Chunking is not guaranteed single-turn — a packed chunk holding both roles is 'mixed'."""
    text = "User: how much is the bus?\nAssistant: usually around 3000 JPY."
    sid, kind = parse_speaker(text)
    assert kind == "mixed" and sid == speaker_row_for("mixed").speaker_id


def test_parse_no_marker_is_none():
    assert parse_speaker("Just a plain note, no role markers here.") == (None, None)
    assert parse_speaker("") == (None, None)
    assert parse_speaker(None) == (None, None)


def test_parse_tolerates_header_and_human_ai_aliases():
    # a re-prefixed chat-session header line is not a role marker; Human/AI map onto the buckets
    assert detect_roles("[chat session — 2024]\nHuman: hi\nAI: hello") == {"human", "assistant"}
    assert parse_speaker("[chat session]\nUser: hi")[1] == "human"


def test_speaker_ids_stable_and_distinct():
    u = speaker_row_for("human").speaker_id
    a = speaker_row_for("assistant").speaker_id
    m = speaker_row_for("mixed").speaker_id
    assert len({u, a, m}) == 3
    assert speaker_row_for("human").speaker_id == u   # deterministic across calls


# --------------------------------------------------------------------------- #
# 2. registry upsert + persistence
# --------------------------------------------------------------------------- #
def _fresh_store():
    store = GraphStore(Config.default(), path=os.path.join(tempfile.mkdtemp(), "kg.db"))
    store._init_db()
    return store


def test_registry_upsert_idempotent():
    store = _fresh_store()
    sid = ensure_speaker(store, "human")
    assert store._dirty_speakers == {sid}
    store.save()
    assert not store._dirty_speakers
    ensure_speaker(store, "human")                 # re-assert identical row
    assert store._dirty_speakers == set()          # no-op — nothing re-marked dirty
    row = store.get_speaker(sid)
    assert row.kind == "human" and row.canonical_name == "user" and "User" in row.aliases


def test_registry_persists_and_reloads():
    store = _fresh_store()
    ensure_speaker(store, "human")
    ensure_speaker(store, "assistant")
    store.save()
    reopened = GraphStore.open(store.path)
    kinds = {r.kind for r in reopened.speakers.values()}
    assert kinds == {"human", "assistant"}


def test_load_tolerates_pre_feature_db_without_speakers_table():
    """A store written before the feature has no `speakers` table — loading must not raise."""
    store = _fresh_store()
    store.add_node(episode_node("ep_x", modality=Modality.TEXT, source_ref="s",
                                raw_text="User: hi", content_hash="h", ts="2024-01-01"))
    store.save()
    import sqlite3
    con = sqlite3.connect(store.path)
    con.execute("DROP TABLE speakers")
    con.commit()
    con.close()
    reopened = GraphStore.open(store.path)          # must not raise
    assert reopened.speakers == {}
    assert reopened.get_node("ep_x") is not None


# --------------------------------------------------------------------------- #
# helpers to build a controlled attribution store
# --------------------------------------------------------------------------- #
def _episode(store, eid, raw, ts):
    node = episode_node(eid, modality=Modality.TEXT, source_ref="s", raw_text=raw,
                        content_hash="h_" + eid, ts=ts)
    stamp_episode(store, node)
    store.add_node(node)
    return node


def _attrib_store():
    """me --went_to--> park (asserted by an ASSISTANT chunk, confirmed by a USER chunk →
    user-grounded) and me --flies_to--> Chicago (asserted by the ASSISTANT chunk only →
    assistant-grounded)."""
    cfg = Config.default()
    cfg.embedder = "st"
    store = GraphStore(cfg, path=os.path.join(tempfile.mkdtemp(), "kg.db"))
    store._init_db()
    canon = Canonicalizer(store, get_embedder(cfg), cfg)
    for nid, name in [("e_me", "me"), ("e_park", "the park"), ("e_chi", "Chicago")]:
        store.add_node(entity_node(nid, name=name, etype=EntityType.CONCEPT, ts="t"))
    _episode(store, "ep_asst", "Assistant: you might enjoy the park; many people fly to Chicago.",
             "2024-01-01T00:00:00+00:00")
    _episode(store, "ep_user", "User: I went to the park on Saturday.",
             "2024-01-02T00:00:00+00:00")
    went = canon.resolve_relation("went_to")
    flies = canon.resolve_relation("flies_to")
    # went_to park: assistant asserts, user re-asserts (confirm → confirmed_by=[ep_user])
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went, status="asserted",
               at="2024-01-01", episode_id="ep_asst")
    apply_fact(store, src="e_me", dst="e_park", rel_tag=went, status="asserted",
               at="2024-01-02", episode_id="ep_user")
    # flies_to Chicago: assistant only
    apply_fact(store, src="e_me", dst="e_chi", rel_tag=flies, status="asserted",
               at="2024-01-01", episode_id="ep_asst")
    return store, cfg


def _edge(store, src, dst):
    return list(store.g.get_edge_data(src, dst).values())[0]


# --------------------------------------------------------------------------- #
# 3. attribution — any-user reduction over episode_id ∪ confirmed_by
# --------------------------------------------------------------------------- #
def test_attribution_assistant_only_is_assistant_grounded():
    store, _cfg = _attrib_store()
    kinds = asserted_by(store, _edge(store, "e_me", "e_chi"))
    assert kinds == ["assistant"]
    assert is_assistant_only(kinds) is True


def test_attribution_user_echo_is_user_grounded():
    """episode_id (assistant) ∪ confirmed_by (user) → any-user wins → NOT assistant-only."""
    store, _cfg = _attrib_store()
    data = _edge(store, "e_me", "e_park")
    assert set(data.get("confirmed_by") or []) == {"ep_user"}     # the user re-assertion
    kinds = asserted_by(store, data)
    assert kinds == ["assistant", "human"]
    assert is_assistant_only(kinds) is False


def test_attribution_mixed_counts_as_containing_human():
    store = _fresh_store()
    _episode(store, "ep_mix", "User: how much?\nAssistant: about 3000 JPY.", "2024-01-01")
    data = {"episode_id": "ep_mix", "confirmed_by": []}
    assert asserted_by(store, data) == ["mixed"]
    assert is_assistant_only(["mixed"]) is False                  # mixed never marked


def test_attribution_unknown_provenance_is_unmarked():
    store = _fresh_store()
    _episode(store, "ep_plain", "A plain note with no role markers.", "2024-01-01")
    data = {"episode_id": "ep_plain", "confirmed_by": []}
    assert asserted_by(store, data) == []                         # no speaker stamped
    assert is_assistant_only([]) is False                        # conservative: don't discount


# --------------------------------------------------------------------------- #
# 4. marker rendering — assistant-only marked, user-echoed UNmarked (knob-gated)
# --------------------------------------------------------------------------- #
def _build_ctx(store, cfg):
    res = RetrievalResult(query="where do I go?", mode="local",
                          objects=[("ep_user", 1.0), ("ep_asst", 0.9)],
                          subgraph={"e_me", "e_park", "e_chi"})
    return ContextBuilder(store, cfg).build(res)


def test_marker_on_assistant_line_only_when_knob_on():
    store, cfg = _attrib_store()
    on = replace(cfg, speaker_attribution=True)
    _ep, _facts, blob = _build_ctx(store, on)
    lines = [ln for ln in blob.splitlines() if "-->" in ln]
    chi = next(ln for ln in lines if "Chicago" in ln)
    park = next(ln for ln in lines if "park" in ln)
    assert "[assistant]" in chi                    # assistant-only fact is flagged
    assert "[assistant]" not in park               # user-echoed fact is NOT flagged


def test_marker_absent_when_knob_off_and_context_byte_identical():
    store, cfg = _attrib_store()
    off = replace(cfg, speaker_attribution=False)
    on = replace(cfg, speaker_attribution=True)
    _e0, _f0, blob_off = _build_ctx(store, off)
    _e1, _f1, blob_on = _build_ctx(store, on)
    assert "[assistant]" not in blob_off
    # the ONLY difference the knob makes is adding " [assistant]" suffixes — strip them and the
    # on-context is byte-identical to the off-context (nothing else in the blob moved).
    assert blob_on.replace(" [assistant]", "") == blob_off


# --------------------------------------------------------------------------- #
# asserted_by rides the structured row (for programmatic/agent filtering)
# --------------------------------------------------------------------------- #
def test_asserted_by_on_fact_row():
    from kg.facts import FactLine
    store, _cfg = _attrib_store()
    row = FactLine.from_edge(store, "e_me", "e_chi", _edge(store, "e_me", "e_chi")).to_row()
    assert row["asserted_by"] == ["assistant"]


# --------------------------------------------------------------------------- #
# 5. knob OFF ⇒ system prompt byte-identical (fake answer client captures the prompt)
# --------------------------------------------------------------------------- #
class _CaptureFake:
    def __init__(self):
        self.chat = self
        self.completions = self
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        tc = types.SimpleNamespace(id="c0", function=types.SimpleNamespace(
            name="submit_answer",
            arguments=json.dumps({"answer": "ok", "citations": [], "events": []})))
        message = types.SimpleNamespace(content=None, tool_calls=[tc])
        choice = types.SimpleNamespace(message=message, finish_reason="tool_calls")
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        return types.SimpleNamespace(choices=[choice], usage=usage)


def _sys_prompt_for(speaker_attribution: bool) -> str:
    store, cfg = _attrib_store()
    cfg = replace(cfg, speaker_attribution=speaker_attribution)
    fake = _CaptureFake()
    ans = OpenAIAnswerer(store, cfg, ContextBuilder(store, cfg), client=fake)
    res = RetrievalResult(query="where do I go?", mode="local",
                          objects=[("ep_user", 1.0)], subgraph={"e_me", "e_park", "e_chi"})
    ans.answer(res)
    return fake.calls[0]["messages"][0]["content"]


def test_prompt_byte_identical_when_off():
    assert _sys_prompt_for(False) == _RAG_SYS


def test_prompt_gains_speaker_rule_when_on():
    on = _sys_prompt_for(True)
    assert on == _RAG_SYS + _SPEAKER_RULE
    assert "[assistant]" in on and "isn't available" in on


# --------------------------------------------------------------------------- #
# 6. backfill — idempotent + incremental on a scripted store; cache-key unchanged
# --------------------------------------------------------------------------- #
def _E(name):
    return ExtractedEntity(name=name, type=EntityType.CONCEPT)


def _R(src, tgt, label):
    return ExtractedRelation(source=src, target=tgt, labels=[label],
                             provenance=Provenance.EXTRACTED, confidence=0.95,
                             status="asserted")


def _chat_graph(cfg):
    """A store whose sessions are chat transcripts → chunked into User:/Assistant: turns."""
    text = ("[chat session — 2024-01-01]\n"
            "User: I'm planning a trip to Tokyo and want to budget the airport transfer.\n"
            "Assistant: The Airport Limousine Bus is typically 3000 JPY; a taxi can be much more.\n"
            "User: Great, I'll take the bus then. I already booked the Park Hyatt.\n"
            "Assistant: Nice choice — the Park Hyatt is in Shinjuku.")
    item = CorpusItem(id="s1", modality="text", source_ref="chat/s1", title="trip",
                      text=text, created_at="2024-01-01T00:00:00+00:00")
    scripted = ScriptedExtractor({})   # extraction content irrelevant to speaker stamping
    g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), cfg)
    g.extractor = scripted
    g.ingest([item])
    g.save()
    return g


def test_backfill_idempotent_and_incremental():
    cfg = Config.default()
    cfg.embedder = "st"
    cfg.chunking = "turns"
    g = _chat_graph(cfg)
    # simulate a pre-feature store: strip the stamps ingest just wrote
    for n in g.store.nodes_of_type(NodeType.EPISODE):
        n.speaker_id = None
    r1 = backfill_speakers(g.store)
    assert r1["stamped"] > 0 and r1["changed"] == r1["stamped"]   # first pass stamps all
    r2 = backfill_speakers(g.store)
    assert r2["changed"] == 0                                     # idempotent second pass
    assert r2["stamped"] == r1["stamped"]
    # every chunk got a speaker_id, and the registry has the roles the transcript used
    kinds = {row.kind for row in g.store.speakers.values()}
    assert kinds and kinds <= {"human", "assistant", "mixed"}


def test_backfill_does_not_change_ingest_cache_key():
    """speaker_attribution is NOT an ingest-cache field, and backfill touches only payloads +
    the speakers table — so the cached benchmark stores stay valid ($0 in place)."""
    assert "speaker_attribution" not in INGEST_RELEVANT_FIELDS
    sessions = [CorpusItem(id="s1", modality="text", source_ref="c", title="t",
                           text="User: hi\nAssistant: hello", created_at="2024-01-01")]
    base = Config.default()
    key_off = ingest_cache_key("inst", sessions, base)
    key_on = ingest_cache_key("inst", sessions, replace(base, speaker_attribution=True))
    assert key_off == key_on                    # knob does not re-key → no paid re-ingest


# --------------------------------------------------------------------------- #
# 7. backfill on a COPY of a real cached store (skips if none present)
# --------------------------------------------------------------------------- #
def test_backfill_on_real_cached_store_copy():
    import glob
    from kg.ingest_cache import _sqlite_copy
    cands = sorted(glob.glob("store/cache/*.db")) + \
        sorted(glob.glob("runs/sample-datefix-events-1/*.db"))
    if not cands:
        pytest.skip("no real cached store available")
    work = os.path.join(tempfile.mkdtemp(), "copy.db")
    _sqlite_copy(cands[0], work)
    g = KnowledgeGraph.open(work, Config.default())
    r1 = g.backfill_speakers()
    g.save()
    assert r1["stamped"] >= 0                    # runs cleanly on a real store
    g2 = KnowledgeGraph.open(work, Config.default())
    r2 = g2.backfill_speakers()
    assert r2["changed"] == 0                    # idempotent on reopen (stamps persisted)
