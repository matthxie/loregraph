"""Extraction-layer date handling (the upstream feed for the event_facts write path).

Covered behaviors, all offline (fake client / scripted extractor, no API key touched):
  1. the extraction request carries the episode's reference date (item.created_at) so
     stated partial/relative dates can resolve to ISO valid_from/valid_to;
  2. the deterministic filter (_filter_date_terms) drops pure-date AND bare-numeric
     entities, pure-date tags, and relations with such endpoints — salvaging a
     parseable date into a surviving CLEAN sibling relation's valid_from — and counts
     what it removed (Extraction.date_drops);
  3. the filter is applied by the INGESTOR (config.ingest_date_filter) to every
     backend, scripted/local included — not per-extractor — and knob-off writes are
     byte-identical to unfiltered extraction.
"""
import json
import os
import tempfile
from types import SimpleNamespace

import pytest

import kg.graph as kg_graph
from kg.config import Config
from kg.corpus import CorpusItem
from kg.extractors import (Extraction, ExtractedEntity, ExtractedRelation,
                           OpenAIExtractor, ScriptedExtractor, _filter_date_terms,
                           _is_bare_numeric, _is_pure_date, _resolve_date_iso,
                           extract_text_sectioned)
from kg.graph import KnowledgeGraph
from kg.models import EdgeType, EntityType, NodeType
from kg.store import fact_active


@pytest.fixture(autouse=True)
def _no_live_extractor(monkeypatch):
    """kg auto-loads the project .env (real key) and KnowledgeGraph.__init__ eagerly
    builds a live extractor. Drop the key and patch get_extractor so every graph in
    this file is scripted — no OpenAI call is ever made."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(kg_graph, "get_extractor",
                        lambda config: ScriptedExtractor({}))


# --------------------------------------------------------------------------- #
# fake client: captures every request, answers with a canned emit_graph payload
# --------------------------------------------------------------------------- #
def _tool_msg(payload: dict):
    fn = SimpleNamespace(name="emit_graph", arguments=json.dumps(payload))
    msg = SimpleNamespace(tool_calls=[SimpleNamespace(function=fn)])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


class _FakeClient:
    def __init__(self, payload: dict):
        self.calls: list[dict] = []
        self._payload = payload
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _tool_msg(self._payload)


def _extractor(monkeypatch, payload: dict) -> tuple[OpenAIExtractor, _FakeClient]:
    fake = _FakeClient(payload)
    monkeypatch.setattr("kg.extractors.make_client", lambda: fake)
    cfg = Config()
    cfg.llm_model = "fake-model"      # explicit → resolve_model never probes a provider
    cfg.reflexion = False             # one call per extract_text, easy capture
    return OpenAIExtractor(cfg), fake


# --------------------------------------------------------------------------- #
# 1. pure-date / bare-numeric detection + resolution helpers
# --------------------------------------------------------------------------- #
def test_pure_date_detection():
    for s in ("January 2nd", "February 1st", "january 2nd, 2023", "2 January 2023",
              "2023", "2024-03-05", "next Tuesday", "Tuesday", "last March",
              "yesterday", "last week", "last spring", "the 2nd of January"):
        assert _is_pure_date(s), s
    # conservative: names/places/bare months/seasons are NEVER swallowed
    for s in ("St. Mary's Church", "May", "August", "Johnson and Johnson", "spring",
              "cathedral", "March on Washington", ""):
        assert not _is_pure_date(s), s


def test_bare_numeric_detection():
    for s in ("1", "42", "3rd", "$15", "1,500", "20%", "3.14"):
        assert _is_bare_numeric(s), s
    # 4-digit years stay classified as DATES, not bare numerics
    assert not _is_bare_numeric("2023")
    assert _is_pure_date("2023")
    for s in ("Room 101", "Alice", "January 2nd", ""):
        assert not _is_bare_numeric(s), s


def test_date_resolution_against_reference():
    assert _resolve_date_iso("January 2nd", "2024-01-15") == "2024-01-02"
    assert _resolve_date_iso("February 1st", "2024-01-15") == "2024-02-01"
    assert _resolve_date_iso("last March", "2024-01-15") == "2023-03"
    assert _resolve_date_iso("next March", "2024-06-15") == "2025-03"
    assert _resolve_date_iso("March 2023", "") == "2023-03"
    assert _resolve_date_iso("January 2nd, 2023", "") == "2023-01-02"
    assert _resolve_date_iso("2023", "") == "2023"
    assert _resolve_date_iso("2024-03-05", "") == "2024-03-05"
    # never guess: weekdays/relative words, or partial dates with no reference
    assert _resolve_date_iso("next Tuesday", "2024-01-15") == ""
    assert _resolve_date_iso("yesterday", "2024-01-15") == ""
    assert _resolve_date_iso("January 2nd", "") == ""
    assert _resolve_date_iso("1", "2024-01-15") == ""      # bare numeric ≠ a date


# --------------------------------------------------------------------------- #
# 2. the outgoing LLM request carries the reference date + the no-date instructions
# --------------------------------------------------------------------------- #
def test_request_contains_reference_date(monkeypatch):
    ext, fake = _extractor(monkeypatch, {"entities": [], "tags": []})
    ext.extract_text("Attended the service on January 2nd.", ref_date="2024-01-15")
    assert fake.calls, "the extractor never called the (fake) client"
    user = fake.calls[0]["messages"][1]["content"][0]["text"]
    assert "This content is dated 2024-01-15." in user
    assert "never guess" in user
    sys = fake.calls[0]["messages"][0]["content"]
    assert "NEVER entities" in sys and "NEVER a relation endpoint" in sys


def test_request_reference_date_falls_back_to_unknown(monkeypatch):
    ext, fake = _extractor(monkeypatch, {"entities": [], "tags": []})
    ext.extract_text("Some undated note.")
    user = fake.calls[0]["messages"][1]["content"][0]["text"]
    assert "This content is dated unknown." in user


def test_sectioned_path_threads_ref_date():
    seen: list[str] = []

    class _Rec:
        name = "rec"

        def extract_text(self, text, title="", ref_date=""):
            seen.append(ref_date)
            return Extraction()

    extract_text_sectioned(_Rec(), "x" * 50, long_doc_chars=10, ref_date="2024-01-15")
    assert seen and all(rd == "2024-01-15" for rd in seen)
    # ScriptedExtractor keeps protocol parity (ingest passes ref_date to every backend)
    assert ScriptedExtractor({}).extract_text("a", ref_date="2024-01-15") is not None


# --------------------------------------------------------------------------- #
# 3. the filter itself: drop + salvage semantics (direct construction — the same
#    object any backend hands the Ingestor)
# --------------------------------------------------------------------------- #
def test_date_and_numeric_terms_filtered_salvage_prefers_clean_sibling():
    ext = Extraction(
        entities=[ExtractedEntity(name="Alice", type=EntityType.PERSON),
                  ExtractedEntity(name="St. Mary's Church", type=EntityType.PLACE),
                  ExtractedEntity(name="January 2nd", type=EntityType.DATE),
                  ExtractedEntity(name="42", type=EntityType.CONCEPT)],
        tags=["church", "2023", "faith"],
        relations=[
            ExtractedRelation(source="Alice", target="St. Mary's Church",
                              labels=["attended"]),
            # the live-run garbage sibling: bare-digit target — dropped, never salvaged to
            ExtractedRelation(source="St. Mary's Church", target="1",
                              labels=["attended"]),
            ExtractedRelation(source="St. Mary's Church", target="January 2nd",
                              labels=["attended"]),
        ])
    out = _filter_date_terms(ext, "2024-01-15")
    assert [e.name for e in out.entities] == ["Alice", "St. Mary's Church"]
    assert out.tags == ["church", "faith"]                    # pure-date tag dropped
    assert len(out.relations) == 1
    r = out.relations[0]
    assert (r.source, r.target) == ("Alice", "St. Mary's Church")
    assert r.valid_from == "2024-01-02"     # salvaged onto the CLEAN sibling
    assert out.date_drops == 5              # 2 entities + 1 tag + 2 relations


def test_salvage_refuses_degenerate_sibling():
    # the only sibling sharing the anchor is a self-loop — salvage must not touch it
    ext = Extraction(relations=[
        ExtractedRelation(source="St. Mary's Church", target="St. Mary's Church",
                          labels=["attended"]),
        ExtractedRelation(source="St. Mary's Church", target="January 2nd",
                          labels=["attended"]),
    ])
    out = _filter_date_terms(ext, "2024-01-15")
    assert len(out.relations) == 1
    assert out.relations[0].valid_from == ""        # refused: degenerate (self-loop)
    assert out.date_drops == 1


def test_filter_respects_existing_valid_from_and_double_date_edges():
    ext = Extraction(
        entities=[ExtractedEntity(name="cathedral", type=EntityType.PLACE)],
        relations=[
            ExtractedRelation(source="me", target="cathedral", labels=["visited"],
                              valid_from="2023-05-01"),
            ExtractedRelation(source="cathedral", target="February 1st",
                              labels=["reflect"]),
            ExtractedRelation(source="2023", target="January 2nd", labels=["before"]),
        ])
    out = _filter_date_terms(ext, "2024-01-15")
    assert len(out.relations) == 1
    assert out.relations[0].valid_from == "2023-05-01"   # stated bound never overwritten
    assert out.date_drops == 2


# --------------------------------------------------------------------------- #
# 4. ingest choke point: EVERY backend is filtered when the knob is on — here a
#    SCRIPTED extractor (never passes through any LLM prompt/parse path)
# --------------------------------------------------------------------------- #
_TXT = "Alice attended St. Mary's Church on January 2nd."


def _scripted_date_table() -> dict[str, Extraction]:
    # fresh objects each call: the filter mutates the Extraction in place
    return {_TXT: Extraction(
        entities=[ExtractedEntity(name="Alice", type=EntityType.PERSON),
                  ExtractedEntity(name="St. Mary's Church", type=EntityType.PLACE),
                  ExtractedEntity(name="January 2nd", type=EntityType.DATE)],
        tags=["church", "faith"],
        relations=[
            ExtractedRelation(source="Alice", target="St. Mary's Church",
                              labels=["attended"]),
            ExtractedRelation(source="St. Mary's Church", target="January 2nd",
                              labels=["attended"]),
        ])}


def _ingest_scripted(ingest_date_filter: bool):
    cfg = Config.default()
    cfg.embedder = "st"     # real local bge — deterministic, free, no key/network once cached
    cfg.ingest_date_filter = ingest_date_filter
    g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), cfg)
    g.extractor = ScriptedExtractor(_scripted_date_table())
    report = g.ingest([CorpusItem(id="d01", modality="text", source_ref="synthetic/d01",
                                  title="church visit", text=_TXT,
                                  created_at="2024-01-15T00:00:00+00:00")])
    return g, report


def _entity_names(g) -> set[str]:
    return {n.name for n in g.store.nodes_of_type(NodeType.ENTITY)}


def test_ingest_filters_scripted_extraction_with_knob_on():
    g, report = _ingest_scripted(ingest_date_filter=True)
    names = _entity_names(g)
    assert "St. Mary's Church" in names and "Alice" in names
    assert "January 2nd" not in names                 # date never became an entity
    assert any("date-term filter" in n for n in report.notes)   # observable, not silent
    # the junk edge's date was salvaged into the real relation's valid_from
    alice = next(n.id for n in g.store.nodes_of_type(NodeType.ENTITY) if n.name == "Alice")
    facts = [(g.store.get_node(d["rel_tag"]).name, g.store.get_node(nbr).name, d)
             for nbr, d in g.store.neighbors(alice, etypes={EdgeType.RELATED_TO},
                                             direction="both") if fact_active(d, None)]
    assert [(p, o) for p, o, _ in facts] == [("attended", "St. Mary's Church")]
    assert facts[0][2]["valid_at"] == "2024-01-02"


def test_ingest_knob_off_writes_unfiltered_extraction():
    g, report = _ingest_scripted(ingest_date_filter=False)
    names = _entity_names(g)
    # byte-identical to pre-knob behavior for scripted/local backends: the date
    # entity and the date-endpoint relation land in the graph, and no filter note
    assert "January 2nd" in names
    assert not any("date-term filter" in n for n in report.notes)
    church = next(n.id for n in g.store.nodes_of_type(NodeType.ENTITY)
                  if n.name == "St. Mary's Church")
    targets = {g.store.get_node(nbr).name
               for nbr, d in g.store.neighbors(church, etypes={EdgeType.RELATED_TO},
                                               direction="both")}
    assert "January 2nd" in targets                   # junk edge preserved as today


def test_llm_extractor_no_longer_filters_inline(monkeypatch):
    """The per-extractor filter was removed: the LLM parse path now hands date junk
    through untouched (the Ingestor is the single choke point)."""
    payload = {
        "entities": [{"name": "January 2nd", "type": "date", "category": "thing"}],
        "tags": ["2023"],
        "relations": [{"source": "St. Mary's Church", "target": "January 2nd",
                       "labels": ["attended"]}],
    }
    ext, _ = _extractor(monkeypatch, payload)
    out = ext.extract_text("…", ref_date="2024-01-15")
    assert [e.name for e in out.entities] == ["January 2nd"]
    assert out.tags == ["2023"]
    assert len(out.relations) == 1 and out.date_drops == 0
