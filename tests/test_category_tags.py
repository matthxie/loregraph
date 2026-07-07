"""Smoke tests for the extraction-side event/category changes (offline, no key)."""
from __future__ import annotations

import os
import tempfile

import pytest

from kg import Config, KnowledgeGraph
from kg.corpus import CorpusItem
from kg.cues import cue_kinds
from kg.extractors import (Extraction, ExtractedEntity, ExtractedRelation,
                           ScriptedExtractor, _parse_tool_payload)
from kg.models import EdgeType


@pytest.fixture(autouse=True)
def _no_live_extractor(monkeypatch):
    import kg.graph as _graph
    monkeypatch.setattr(_graph, "get_extractor", lambda config: ScriptedExtractor({}))


def test_user_action_cue_fires():
    assert "user_action" in cue_kinds("You've already subscribed to Architectural Digest.")
    assert "user_action" in cue_kinds("I attended a yoga class this morning.")
    assert "user_action" not in cue_kinds("The weather is nice today in Paris.")


def test_parse_payload_category():
    ext = _parse_tool_payload({
        "entities": [{"name": "Architectural Digest", "type": "org", "category": "Magazine"},
                     {"name": "yoga", "type": "concept", "category": "yoga"}],
        "tags": ["home decor"],
    })
    assert ext.entities[0].category == "magazine"      # lowercased
    assert ext.entities[1].category == ""              # self-category dropped


def test_merge_adopts_missing_category():
    base = Extraction(entities=[ExtractedEntity(name="Zelda")])
    other = Extraction(entities=[ExtractedEntity(name="zelda", category="video game")])
    base.merge(other)
    assert base.entities[0].category == "video game"


def test_ingest_writes_category_tag_and_action_fact():
    text = "I subscribed to Architectural Digest last week."
    table = {text.strip().lower(): Extraction(
        entities=[ExtractedEntity(name="me"),
                  ExtractedEntity(name="Architectural Digest", category="magazine")],
        tags=["home decor"],
        relations=[ExtractedRelation(source="me", target="Architectural Digest",
                                     labels=["subscribed_to"])],
    )}
    c = Config.default()
    c.embedder = "st"
    path = os.path.join(tempfile.mkdtemp(), "kg.db")
    g = KnowledgeGraph.open(path, c)
    g.extractor = ScriptedExtractor(table)
    rep = g.ingest([CorpusItem(id="s1", modality="text", source_ref="u/s1", text=text)])
    assert rep.ingested == 1 and rep.facts >= 1
    ep = g.store.get_node("ep_s1")
    assert "magazine" in [t.lower() for t in ep.tags]
    tag_ids = [nbr for nbr, _ in g.store.neighbors("ep_s1", etypes={EdgeType.TAGGED_AS},
                                                   direction="out")]
    assert any(g.store.get_node(t).name == "magazine" for t in tag_ids)
