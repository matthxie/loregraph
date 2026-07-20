"""URL ingestion (described-media LINK path) wiring tests.

A saved bare URL becomes a LINK episode: text=None, the LLM-authored `description` is the
embedding/retrieval surface, and extraction is SUBJECT-SCOPED — the record is about the
page's primary subject, not an inventory of every entity named on it. Fully hermetic:
KG_LINK_FETCH=0 disables the network fetch, and a ScriptedExtractor keyed by URL stands in
for the live subject-scoped LLM (same policy as tests/test_temporal.py). Embeddings use the
real local bge model. Run: python -m pytest tests/test_url_ingest.py -q
"""
from __future__ import annotations

import tempfile

import pytest

from kg.engine import Engine, NoteInput
from kg.extractors import (Extraction, ExtractedEntity, ExtractedRelation,
                           ScriptedExtractor, fetch_page)
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
    assert ep["text"] == ""                                  # LINK has no raw_text surface
    node = eng._g.store.get_node(res.episode_id)
    assert node.modality is Modality.LINK
    assert node.raw_text is None
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
