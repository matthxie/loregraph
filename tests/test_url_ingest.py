"""URL ingestion (LINK path) wiring tests.

A saved bare URL becomes a LINK episode: the byte-exact capture is preserved as raw_text
(the display fallback when title resolution fails), the resolved page title/description is
the embedding/retrieval surface, and extraction is SUBJECT-SCOPED — the record is about the
page's primary subject, not an inventory of every entity named on it. Fully hermetic:
KG_LINK_FETCH=0 disables the network fetch, and a ScriptedExtractor keyed by URL stands in
for the live subject-scoped LLM (same policy as tests/test_temporal.py). Embeddings use the
real local bge model. Run: python -m pytest tests/test_url_ingest.py -q
"""
from __future__ import annotations

import tempfile
import types

import pytest

from kg.corpus import CorpusItem
from kg.engine import Engine, NoteInput
from kg.extractors import (Extraction, ExtractedEntity, ExtractedRelation,
                           ScriptedExtractor, fetch_page)
from kg.ingest import Ingestor, _sha256
from kg.models import EdgeType, EntityType, Modality, NodeType, Provenance


CARD_URL = "https://www.cardco.com/premium-card"

# The card-company scenario from the brief: the page name-drops six banks but is ABOUT the
# card company. A subject-scoped extractor keeps only CardCo + its one concrete relation
# (issued_by BankX), never the passing-mention banks.
_CARD_RECORD = Extraction(
    entities=[ExtractedEntity("CardCo", EntityType.ORG),
              ExtractedEntity("BankX", EntityType.ORG)],
    tags=["credit cards", "rewards", "fintech"],
    relations=[ExtractedRelation(source="CardCo", target="BankX",
                                 labels=["issued_by"], provenance=Provenance.EXTRACTED)],
    description="CardCo is a premium rewards credit card issued by BankX.",
    source_text="CardCo premium card. Compare against Chase, Citi, Amex, Wells Fargo, "
                "Capital One and Barclays. CardCo is issued by BankX. Apply today.",
    page_title="CardCo Premium — Rewards Credit Card",
)


@pytest.fixture(autouse=True)
def _no_fetch(monkeypatch):
    """Hermetic: never touch the network for a URL fetch."""
    monkeypatch.setenv("KG_LINK_FETCH", "0")


def _open():
    return Engine.open(tempfile.mkdtemp(), {"kind": "mock"})


def _link_engine():
    eng = _open()
    eng._g.extractor = ScriptedExtractor({CARD_URL: _CARD_RECORD})
    return eng


# --------------------------------------------------------------------------- #
# fetch helper stays offline under the switch
# --------------------------------------------------------------------------- #
def test_fetch_page_disabled_returns_only_url_and_domain():
    sig = fetch_page(CARD_URL)
    assert sig["url"] == CARD_URL and sig["domain"] == "www.cardco.com"
    assert sig["body"] == "" and sig["title"] == ""


# --------------------------------------------------------------------------- #
# bare URL → LINK episode, described-media surface
# --------------------------------------------------------------------------- #
def test_bare_url_becomes_link_episode():
    eng = _link_engine()
    res = eng.ingest(NoteInput(text=CARD_URL, created_at="2026-07-10T00:00:00Z"))
    ep = eng.episode(res.episode_id)
    assert ep["modality"] == "link"
    assert ep["text"] == CARD_URL                # the capture survives as raw_text
    node = eng._g.store.get_node(res.episode_id)
    assert node.modality is Modality.LINK
    assert node.raw_text == CARD_URL
    assert node.source_ref == CARD_URL
    # (a) the description names the primary subject and is the retrieval surface
    assert ep["description"] == _CARD_RECORD.description
    # derive_title prefers the fetched page title over the description
    assert ep["title"] == "CardCo Premium — Rewards Credit Card"
    eng.close()


def test_subject_scoped_no_passing_mentions():
    eng = _link_engine()
    res = eng.ingest(NoteInput(text=CARD_URL, created_at="2026-07-10T00:00:00Z"))
    ep = eng.episode(res.episode_id)
    ents = {e.lower() for e in ep["entities"]}
    # (b) no free-floating passing-mention banks
    for noise in ("chase", "citi", "amex", "wells fargo", "capital one", "barclays"):
        assert noise not in ents
    # only the subject (and its concrete relation partner) survive
    assert "cardco" in ents
    eng.close()


def test_secondary_entity_only_in_a_real_relation():
    eng = _link_engine()
    res = eng.ingest(NoteInput(text=CARD_URL, created_at="2026-07-10T00:00:00Z"))
    # (c) BankX is present because a concrete relation ties it to the subject
    facts = eng.facts("CardCo")
    assert facts["resolved"]
    preds = {(f["source"], f["predicate"], f["target"]) for f in facts["facts"]}
    assert ("CardCo", "issued_by", "BankX") in preds
    eng.close()


# --------------------------------------------------------------------------- #
# SOURCE provenance node (un-rankable) + PART_OF edge
# --------------------------------------------------------------------------- #
def test_source_provenance_node_wired():
    eng = _link_engine()
    res = eng.ingest(NoteInput(text=CARD_URL, created_at="2026-07-10T00:00:00Z"))
    store = eng._g.store
    ep_id = res.episode_id
    parents = list(store.neighbors(ep_id, etypes={EdgeType.PART_OF}, direction="out"))
    assert len(parents) == 1
    src_id, _d = parents[0]
    src = store.get_node(src_id)
    assert src.ntype is NodeType.SOURCE
    assert src.raw_text == _CARD_RECORD.source_text        # full body preserved verbatim
    eng.close()


# --------------------------------------------------------------------------- #
# raw capture preservation (the "Untitled input" fix)
# --------------------------------------------------------------------------- #
class _RecordingExtractor:
    """Wraps an extractor and records which extract_* method served each item."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[str] = []
        self.name = inner.name
        self.meter = inner.meter

    def __getattr__(self, attr):
        fn = getattr(self._inner, attr)
        if not callable(fn):
            return fn

        def wrapped(*a, **kw):
            self.calls.append(attr)
            return fn(*a, **kw)
        return wrapped


def test_bare_url_routes_through_extract_url_not_extract_text():
    eng = _open()
    rec = _RecordingExtractor(ScriptedExtractor({CARD_URL: _CARD_RECORD}))
    eng._g.extractor = rec
    eng.ingest(NoteInput(text=CARD_URL, created_at="2026-07-10T00:00:00Z"))
    assert "extract_url" in rec.calls
    assert "extract_text" not in rec.calls
    eng.close()


def test_link_capture_is_byte_exact_and_source_ref_is_stripped():
    eng = _link_engine()
    padded = f"  {CARD_URL}\n"                   # still a bare URL after strip
    res = eng.ingest(NoteInput(text=padded, created_at="2026-07-10T00:00:00Z"))
    node = eng._g.store.get_node(res.episode_id)
    assert node.modality is Modality.LINK
    assert node.raw_text == padded               # byte-exact capture
    assert node.source_ref == CARD_URL           # normalized URL for fetching
    assert eng.episode(res.episode_id)["text"] == padded
    eng.close()


def test_failed_metadata_leaves_url_as_display_fallback():
    # URL absent from the scripted table → empty extraction (the fetch-failed / sparse
    # page case): no title, no description — but the URL survives as raw_text.
    dead = "https://dead.example.com/gone"
    eng = _link_engine()
    res = eng.ingest(NoteInput(text=dead, created_at="2026-07-11T00:00:00Z"))
    ep = eng.episode(res.episode_id)
    assert ep["modality"] == "link"
    assert not ep["title"]
    assert ep["text"] == dead
    eng.close()


def test_text_containing_url_stays_text_modality():
    eng = _link_engine()
    mixed = f"Read {CARD_URL} before the meeting"
    res = eng.ingest(NoteInput(text=mixed, created_at="2026-07-10T00:00:00Z"))
    node = eng._g.store.get_node(res.episode_id)
    assert node.modality is Modality.TEXT
    assert node.raw_text == mixed                # complete input unchanged
    eng.close()


def test_link_embed_surface_prefers_page_content_over_url():
    item = CorpusItem(id="x", modality="link", source_ref=CARD_URL, text=CARD_URL)
    surface = Ingestor._embed_surface(types.SimpleNamespace(config=None), item,
                                      _CARD_RECORD)
    assert surface == (f"{_CARD_RECORD.page_title}\n{_CARD_RECORD.description}")
    # no resolved page content → the URL is the fallback surface
    fallback = Ingestor._embed_surface(types.SimpleNamespace(config=None), item,
                                       Extraction())
    assert fallback == CARD_URL


# --------------------------------------------------------------------------- #
# compatibility repair for legacy link episodes (raw_text=None era)
# --------------------------------------------------------------------------- #
def _legacy_strip(eng, ep_id):
    """Rewrite a link episode to the pre-fix on-disk shape: raw_text dropped."""
    node = eng._g.store.get_node(ep_id)
    node.raw_text = None
    eng._g.store.touch_node(ep_id)
    eng._g.save()


def test_repair_restores_url_on_reopen_and_is_idempotent():
    data_dir = tempfile.mkdtemp()
    eng = Engine.open(data_dir, {"kind": "mock"})
    eng._g.extractor = ScriptedExtractor({CARD_URL: _CARD_RECORD})
    res = eng.ingest(NoteInput(text=CARD_URL, created_at="2026-07-10T00:00:00Z"))
    _legacy_strip(eng, res.episode_id)
    eng.close()

    eng2 = Engine.open(data_dir, {"kind": "mock"})   # open-boundary repair fires here
    node = eng2._g.store.get_node(res.episode_id)
    assert node.raw_text == CARD_URL
    assert node.title == _CARD_RECORD.page_title     # resolved title untouched
    assert eng2.episode(res.episode_id)["text"] == CARD_URL
    assert eng2.repair_link_raw_text() == 0          # idempotent: nothing left to fix
    eng2.close()

    eng3 = Engine.open(data_dir, {"kind": "mock"})   # repaired value survives save/reopen
    assert eng3._g.store.get_node(res.episode_id).raw_text == CARD_URL
    eng3.close()


def test_repair_never_overwrites_non_empty_raw_text():
    data_dir = tempfile.mkdtemp()
    eng = Engine.open(data_dir, {"kind": "mock"})
    eng._g.extractor = ScriptedExtractor({CARD_URL: _CARD_RECORD})
    res = eng.ingest(NoteInput(text=CARD_URL, created_at="2026-07-10T00:00:00Z"))
    node = eng._g.store.get_node(res.episode_id)
    node.raw_text = "user-edited capture"
    eng._g.store.touch_node(res.episode_id)
    eng._g.save()
    eng.close()

    eng2 = Engine.open(data_dir, {"kind": "mock"})
    assert eng2._g.store.get_node(res.episode_id).raw_text == "user-edited capture"
    eng2.close()


def test_repaired_url_reaches_listing_and_keyword_search():
    dead = "https://dead.example.com/gone"       # sparse: no title/description to lean on
    data_dir = tempfile.mkdtemp()
    eng = Engine.open(data_dir, {"kind": "mock"})
    eng._g.extractor = ScriptedExtractor({})
    res = eng.ingest(NoteInput(text=dead, created_at="2026-07-11T00:00:00Z"))
    _legacy_strip(eng, res.episode_id)
    eng.close()

    eng2 = Engine.open(data_dir, {"kind": "mock"})
    rows = eng2.episodes_list()["episodes"]
    assert any(r["id"] == res.episode_id and r["text"] == dead for r in rows)
    hits = eng2.search("dead example")["episodes"]
    assert any(h["id"] == res.episode_id for h in hits)
    eng2.close()


def test_legacy_hash_recapture_skips_instead_of_versioning():
    """A store written before the fix hashed link items with EMPTY content. A byte-
    identical re-capture must be recognized as unchanged (hash upgraded in place),
    not version-appended as a duplicate ep_X_v1."""
    created = "2026-07-10T00:00:00Z"
    eng = _link_engine()
    res = eng.ingest(NoteInput(text=CARD_URL, created_at=created))
    store = eng._g.store
    ep_id = res.episode_id
    nid = ep_id[len("ep_"):]
    # rewrite the episode + hash cache to the legacy (content="") formula
    legacy_h = _sha256("link", "", created, nid)
    node = store.get_node(ep_id)
    node.content_hash = legacy_h
    for h, eid in list(store.hash_cache.items()):
        if eid == ep_id:
            store.hash_cache.pop(h)
    store.add_hash(legacy_h, ep_id)

    res2 = eng.ingest(NoteInput(text=CARD_URL, created_at=created))
    assert res2.skipped
    assert not store.has_node(f"{ep_id}_v1")
    assert node.content_hash != legacy_h         # upgraded to the current formula
    eng.close()
