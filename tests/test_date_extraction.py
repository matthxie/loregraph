"""Extraction-layer date handling (the upstream feed for the event_facts write path).

Three behaviors, all offline (fake client, no API key touched):
  1. the extraction request carries the episode's reference date (item.created_at) so
     stated partial/relative dates can resolve to ISO valid_from/valid_to;
  2. the deterministic post-filter (_filter_date_terms) drops pure-date entities/tags
     and date-endpoint relations — salvaging a parseable date into a surviving sibling
     relation's valid_from — and counts what it removed (Extraction.date_drops);
  3. a normal, date-free extraction passes through untouched.
"""
import json
from types import SimpleNamespace

from kg.config import Config
from kg.extractors import (Extraction, ExtractedEntity, ExtractedRelation,
                           OpenAIExtractor, ScriptedExtractor, _filter_date_terms,
                           _is_pure_date, _resolve_date_iso, extract_text_sectioned)
from kg.models import EntityType


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
# 1. pure-date detection + resolution helpers
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


# --------------------------------------------------------------------------- #
# 2. the outgoing request carries the reference date + the no-date instructions
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
# 3. post-filter: date-target relation dropped + salvaged, date entity/tag dropped
# --------------------------------------------------------------------------- #
def test_date_target_relation_filtered_and_salvaged(monkeypatch):
    payload = {
        "entities": [
            {"name": "me", "type": "person", "category": "person"},
            {"name": "St. Mary's Church", "type": "place", "category": "place"},
            {"name": "January 2nd", "type": "date", "category": "thing"},
        ],
        "tags": ["church", "2023", "faith"],
        "relations": [
            {"source": "me", "target": "St. Mary's Church", "labels": ["attend"]},
            {"source": "St. Mary's Church", "target": "January 2nd", "labels": ["attend"]},
        ],
    }
    ext, _ = _extractor(monkeypatch, payload)
    out = ext.extract_text("I attended St. Mary's Church on January 2nd.",
                           ref_date="2024-01-15")
    assert [e.name for e in out.entities] == ["me", "St. Mary's Church"]
    assert out.tags == ["church", "faith"]                    # pure-date tag dropped
    assert len(out.relations) == 1
    r = out.relations[0]
    assert (r.source, r.target) == ("me", "St. Mary's Church")
    assert r.valid_from == "2024-01-02"          # salvaged from the dropped junk edge
    assert out.date_drops == 3                   # 1 entity + 1 tag + 1 relation


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


def test_normal_extraction_passes_through(monkeypatch):
    payload = {
        "entities": [
            {"name": "Marie Curie", "type": "person", "category": "person"},
            {"name": "polonium", "type": "concept", "category": "thing"},
        ],
        "tags": ["chemistry", "radioactivity"],
        "relations": [
            {"source": "Marie Curie", "target": "polonium", "labels": ["discovered"],
             "valid_from": "1898"},
        ],
        "facts": [{"subject": "Marie Curie", "predicate": "won", "value": 2,
                   "unit": "nobel prizes"}],
    }
    ext, _ = _extractor(monkeypatch, payload)
    out = ext.extract_text("Marie Curie discovered polonium in 1898.",
                           ref_date="2024-01-15")
    assert [e.name for e in out.entities] == ["Marie Curie", "polonium"]
    assert out.tags == ["chemistry", "radioactivity"]
    assert len(out.relations) == 1 and out.relations[0].valid_from == "1898"
    assert len(out.facts) == 1
    assert out.date_drops == 0
