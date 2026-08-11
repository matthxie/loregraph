"""Engine facade contract tests (kg/engine.py) — the app-router surface.

Everything here runs with the MOCK provider: no API key, no LLM, deterministic.
Embeddings use the real local bge model (free, offline once cached) — same policy
as the rest of the suite. Run: python -m pytest tests/test_engine.py -q
"""
from __future__ import annotations

import os
import tempfile

import pytest

from kg.engine import Engine, NoteInput
from kg.errors import (EngineError, InvalidInput, NotFound,
                       ProviderUnavailable)
from kg.extractors import (Extraction, ExtractedEntity, ExtractedRelation,
                           ScriptedExtractor)
from kg.models import EntityType, Provenance


def _open(tmp=None, kind="mock", logs=None):
    log = (lambda level, msg: logs.append((level, msg))) if logs is not None else None
    return Engine.open(tmp or tempfile.mkdtemp(), {"kind": kind}, log=log)


NOTE = NoteInput(text="Met Becky in Toronto to plan the Berlin move.",
                 created_at="2026-07-01T10:00:00Z")


def test_open_ingest_retrieve_answer_close(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)   # mock needs NO key
    logs = []
    eng = _open(logs=logs)
    res = eng.ingest(NOTE)
    assert res.episode_id.startswith("ep_") and not res.skipped
    assert res.entities > 0                                # Becky/Toronto/Berlin mentions

    r = eng.retrieve("where is Becky moving?", k=3)
    assert r["episodes"] and r["episodes"][0]["id"] == res.episode_id
    assert "Berlin" in r["episodes"][0]["text"]
    assert r["episodes"][0]["when"].startswith("2026-07-01")
    assert "Berlin" in r["rendered_text"]                  # the ask()-identical prompt blob

    a = eng.answer("where is Becky moving?")
    assert "mock provider" in a["answer"]
    assert a["invalid_citations"] == []                    # canned answer cites nothing

    assert eng.stats()["by_node_type"]["episode"] >= 1
    assert any("engine open" in m for _, m in logs)        # log callback, not stdout
    eng.close()
    eng.close()                                            # idempotent
    with pytest.raises(EngineError):
        eng.stats()


def test_ingest_is_idempotent_by_content_and_created_at():
    eng = _open()
    first = eng.ingest(NOTE)
    again = eng.ingest(NoteInput(text=NOTE.text, created_at=NOTE.created_at))
    assert again.episode_id == first.episode_id and again.skipped
    assert eng.episodes_list()["total"] == 1
    eng.close()


def test_persistence_across_reopen():
    tmp = tempfile.mkdtemp()
    eng = _open(tmp)
    eid = eng.ingest(NOTE).episode_id
    eng.close()
    eng2 = _open(tmp)
    ep = eng2.episode(eid)
    assert ep is not None and "Becky" in ep["text"]
    eng2.close()


def test_delete_episode_and_not_found():
    eng = _open()
    eid = eng.ingest(NOTE).episode_id
    eng.delete_episode(eid)
    assert eng.episode(eid) is None                        # tombstoned = gone from reads
    with pytest.raises(NotFound):
        eng.delete_episode("ep_nope")
    eng.close()


def test_provider_none_answers_raise_but_ingest_works(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    eng = _open(kind="none")
    assert eng.ingest(NOTE).episode_id
    with pytest.raises(ProviderUnavailable):
        eng.answer("anything")
    assert eng.provider_status() == {"kind": "none", "connected": False,
                                     "detail": "no credentials"}
    eng.close()


def test_unsupported_provider_kinds_raise():
    with pytest.raises(ProviderUnavailable):
        _open(kind="grok")
    with pytest.raises(ProviderUnavailable):
        _open(kind="llama")


def test_invalid_input_and_stubs():
    eng = _open()
    with pytest.raises(InvalidInput):
        eng.ingest(NoteInput(text="   ", created_at="2026-07-01"))
    with pytest.raises(InvalidInput):
        eng.retrieve("")
    with pytest.raises(EngineError, match="not implemented"):
        eng.rebuild()
    eng.close()


def test_query_knob_normalization_units():
    """§7.3: mmr_lambda clamps to [0,1] and falls back (never errors) on non-finite/
    unparseable input; since/until accept a bare year (= its Jan-1 start) and compare
    on the 10-char date prefix; garbage dates are InvalidInput."""
    from kg.engine import _norm_event_date, _norm_mmr_lambda
    assert _norm_mmr_lambda(None) is None
    assert _norm_mmr_lambda(0.5) == 0.5
    assert _norm_mmr_lambda(2.5) == 1.0
    assert _norm_mmr_lambda(-1) == 0.0
    assert _norm_mmr_lambda(float("nan")) is None
    assert _norm_mmr_lambda("broad") is None
    assert _norm_event_date(None, "since") is None
    assert _norm_event_date("2025", "since") == "2025-01-01"
    assert _norm_event_date("2026-07-08T09:15:00+00:00", "until") == "2026-07-08"
    with pytest.raises(InvalidInput):
        _norm_event_date("July 8", "since")


def test_retrieve_since_until_event_window():
    eng = _open()
    eng.ingest(NoteInput(text="Met Becky at the climbing gym.",
                         created_at="2026-07-01T10:00:00Z"))
    eng.ingest(NoteInput(text="Becky is moving to Berlin.",
                         created_at="2026-07-02T10:00:00Z"))
    both = eng.retrieve("Becky", k=5)["episodes"]
    assert len(both) == 2
    late = eng.retrieve("Becky", k=5, since="2026-07-02")["episodes"]
    assert late and all(h["when"][:10] >= "2026-07-02" for h in late)
    early = eng.retrieve("Becky", k=5, until="2026-07-01")["episodes"]
    assert early and all(h["when"][:10] <= "2026-07-01" for h in early)
    yeared = eng.retrieve("Becky", k=5, since="2026")["episodes"]  # bare year = Jan-1
    assert len(yeared) == 2
    eng.close()


def test_engine_writes_nothing_to_stdout(capsys):
    eng = _open()
    eng.ingest(NOTE)
    eng.retrieve("Becky", k=2)
    eng.answer("where is Becky?")
    eng.close()
    assert capsys.readouterr().out == ""                   # stdout is the app's channel


# --------------------------------------------------------------------------- #
# profile() — the self anchor's facts plus the graph's vocabulary
# --------------------------------------------------------------------------- #
_MOVE = "I moved to Berlin and started at Foca."
_READ = "I keep reading about distributed systems with Becky."

_PROFILE_SCRIPT = {
    _MOVE: Extraction(
        entities=[ExtractedEntity("Berlin", EntityType.PLACE),
                  ExtractedEntity("Foca", EntityType.ORG)],
        tags=["relocation"],
        relations=[ExtractedRelation(source="I", target="Berlin", labels=["lives_in"],
                                     provenance=Provenance.EXTRACTED, confidence=0.9),
                   ExtractedRelation(source="I", target="Foca", labels=["works_at"],
                                     provenance=Provenance.EXTRACTED, confidence=0.9)],
    ),
    _READ: Extraction(
        entities=[ExtractedEntity("Becky", EntityType.PERSON),
                  ExtractedEntity("distributed systems", EntityType.CONCEPT)],
        tags=["reading"],
    ),
}


def _profile_engine():
    eng = _open()
    eng._g.extractor = ScriptedExtractor(_PROFILE_SCRIPT)   # no LLM, known extractions
    eng.ingest(NoteInput(text=_MOVE, created_at="2026-07-01T10:00:00Z"))
    eng.ingest(NoteInput(text=_READ, created_at="2026-07-02T10:00:00Z"))
    return eng


def test_profile_summarizes_self_facts_entities_and_themes():
    eng = _profile_engine()
    p = eng.profile()

    assert p["as_of"] is None
    rendered = {f["rendered"] for f in p["facts"]}
    assert any("Berlin" in r for r in rendered) and any("Foca" in r for r in rendered)
    assert all(f["status"] == "asserted" for f in p["facts"])
    assert all(f["source"] == "me" for f in p["facts"])     # the self anchor, not a copy

    names = {t: [r["name"] for r in rows] for t, rows in p["entities_by_type"].items()}
    assert names["person"] == ["Becky"]                     # 'me' is the subject, not an entity
    assert names["place"] == ["Berlin"]
    assert "Foca" in names["org"]                           # grouped by entity type, not glyph
    assert "date" not in names and "quantity" not in names  # extraction artifacts stay out
    themes = [t["name"] for t in p["themes"]]
    assert "distributed systems" in themes                  # concept entity reads as a theme
    assert {"relocation", "reading"} <= set(themes)         # …so do tags
    assert all(t["doc_frequency"] >= 1 for t in p["themes"])

    eps = {e["id"] for e in eng.episodes_list()["episodes"]}
    assert p["episode_ids"] and set(p["episode_ids"]) <= eps
    assert p["episode_ids"][0] == p["facts"][0]["episode_id"]   # provenance first

    text = p["rendered_text"]
    assert "Known about the user:" in text and "Berlin" in text
    assert "Recurring themes:" in text and "distributed systems" in text
    eng.close()


def test_profile_top_caps_each_section_and_as_of_is_echoed():
    eng = _profile_engine()
    p = eng.profile(as_of="2026-07-01T12:00:00Z", top=1)
    assert p["as_of"] == "2026-07-01T12:00:00Z"
    assert len(p["facts"]) == 1 and len(p["themes"]) == 1
    assert len(p["episode_ids"]) == 1
    assert all(len(rows) == 1 for rows in p["entities_by_type"].values())
    eng.close()


def test_profile_on_empty_graph_is_not_an_error():
    eng = _open()
    p = eng.profile()
    assert p == {"as_of": None, "facts": [], "entities_by_type": {}, "themes": [],
                 "episode_ids": [], "rendered_text": "No profile information yet."}
    eng.close()
    with pytest.raises(EngineError):
        eng.profile()                                       # closed engine still guards
