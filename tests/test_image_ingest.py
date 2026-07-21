"""Image ingestion wiring tests (the described-media IMAGE path + captioned co-perception).

An ingested image must land as the SAME shape of graph a conversation produces — abstracted
to shared concept/work/person/place nodes and me-anchored relations — never a catalog of
literal visual objects. Two arrival modes are covered:

  * standalone image (media-only, empty text) → IMAGE episode whose LLM-authored `description`
    is the embedding/retrieval surface (text=None);
  * image + caption → CO-PERCEPTION: the caption text pass and the vision pass are BOTH run
    and merged into one episode (caption is the merge base, vision unions in for recall).

Fully hermetic (same policy as tests/test_url_ingest.py / test_temporal.py): a ScriptedExtractor
stands in for the live VLM — extract_image is keyed by the attachment path, extract_text by the
note/caption text — so no OpenAI call is made. Embeddings use the real local bge model.
Run: python -m pytest tests/test_image_ingest.py -q
"""
from __future__ import annotations

import os
import tempfile

from kg.engine import Engine, NoteInput
from kg.extractors import (Extraction, ExtractedEntity, ExtractedRelation,
                           ScriptedExtractor)
from kg.models import EntityType, Modality, NodeType, Provenance


# The cycling scenario from the brief: a prior conversation note establishes the `cycling`
# and `commuter bike` concept nodes; a later image about the same thing must RESOLVE onto
# those existing nodes (the seamless-fit guarantee) rather than mint parallel ones — and it
# must emit those concepts, not literal visual objects (bicycle / trees / sky / wheel).
CYCLING_NOTE = "I got really into cycling this spring and bought a commuter bike."
_CYCLING_NOTE_RECORD = Extraction(
    entities=[ExtractedEntity("cycling", EntityType.CONCEPT),
              ExtractedEntity("commuter bike", EntityType.WORK)],
    tags=["hobby", "spring"],
    relations=[ExtractedRelation(source="me", target="commuter bike",
                                 labels=["bought"], provenance=Provenance.EXTRACTED)],
)

# The vision pass for a photo of the bike on a trail: concepts + a me-anchored relation and a
# retrieval-surface description. NO literal-object entities (bicycle/trees/sky) — the prompt's
# connective-tissue rule says abstract to the concepts that link this to the user's notes.
_BIKE_IMAGE_RECORD = Extraction(
    entities=[ExtractedEntity("cycling", EntityType.CONCEPT),
              ExtractedEntity("commuter bike", EntityType.WORK)],
    tags=["cycling", "outdoors", "trail"],
    relations=[ExtractedRelation(source="me", target="commuter bike",
                                 labels=["rode"], provenance=Provenance.EXTRACTED)],
    description="A commuter bike parked on a wooded cycling trail.",
)

# A caption the user typed alongside the same photo. Carries its OWN first-person signal
# (a place + a me-anchored visit) and, per the design, NO description — the vision pass
# supplies the media surface after the merge.
CAPTION = "Great morning ride through Stanley Park on the new bike."
_CAPTION_RECORD = Extraction(
    entities=[ExtractedEntity("Stanley Park", EntityType.PLACE)],
    tags=["morning ride"],
    relations=[ExtractedRelation(source="me", target="Stanley Park",
                                 labels=["visited"], provenance=Provenance.EXTRACTED)],
)


def _img(dirpath: str, name: str = "bike.jpg") -> str:
    """A real file at a known path — its path is the ScriptedExtractor key for extract_image.
    (The ScriptedExtractor never reads the bytes, but a real path keeps the flow realistic.)"""
    p = os.path.join(dirpath, name)
    with open(p, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    return p


def _engine():
    return Engine.open(tempfile.mkdtemp(), {"kind": "mock"})


# --------------------------------------------------------------------------- #
# (a) standalone image → IMAGE episode, description is the surface
# --------------------------------------------------------------------------- #
def test_standalone_image_becomes_image_episode_with_description_surface():
    d = tempfile.mkdtemp()
    img = _img(d)
    eng = _engine()
    eng._g.extractor = ScriptedExtractor({img: _BIKE_IMAGE_RECORD})
    res = eng.ingest(NoteInput(text="", attachments=[img],
                               created_at="2026-06-01T00:00:00Z"))
    ep = eng.episode(res.episode_id)
    assert ep["modality"] == "image"
    assert ep["text"] == ""                                   # media-only: no raw_text surface
    node = eng._g.store.get_node(res.episode_id)
    assert node.modality is Modality.IMAGE
    assert node.raw_text is None
    assert ep["description"] == _BIKE_IMAGE_RECORD.description  # the retrieval surface
    assert img in ep["media_paths"]
    # graph-aligned: concepts/works, not literal visual objects
    assert "cycling" in {c.lower() for c in ep["concepts"]}
    eng.close()


def test_standalone_image_png_and_no_literal_object_junk_nodes():
    d = tempfile.mkdtemp()
    img = _img(d, "trail.png")                                # sniffs to IMAGE by extension
    eng = _engine()
    eng._g.extractor = ScriptedExtractor({img: _BIKE_IMAGE_RECORD})
    eng.ingest(NoteInput(text="", attachments=[img], created_at="2026-06-01T00:00:00Z"))
    names = {n.name.lower() for n in eng._g.store.nodes_of_type(NodeType.ENTITY)}
    # (d) the connective-tissue concepts are present, the literal visual objects never minted
    assert "commuter bike" in names
    for junk in ("bicycle", "trees", "sky", "wheel", "handlebar", "pavement"):
        assert junk not in names
    eng.close()


# --------------------------------------------------------------------------- #
# (b) image + caption → co-perception merges BOTH passes into one episode
# --------------------------------------------------------------------------- #
def test_captioned_image_co_perceives_caption_and_vision():
    d = tempfile.mkdtemp()
    img = _img(d)
    eng = _engine()
    eng._g.extractor = ScriptedExtractor({CAPTION: _CAPTION_RECORD, img: _BIKE_IMAGE_RECORD})
    res = eng.ingest(NoteInput(text=CAPTION, attachments=[img],
                               created_at="2026-06-02T00:00:00Z"))
    ep = eng.episode(res.episode_id)
    assert ep["modality"] == "image"
    assert ep["text"] == CAPTION                              # caption preserved as raw_text
    # caption left no description → the vision pass supplies the media surface
    assert ep["description"] == _BIKE_IMAGE_RECORD.description
    # the merge unions BOTH passes: the caption's place AND the vision's concept/work
    ents = {e.lower() for e in ep["entities"]}
    concepts = {c.lower() for c in ep["concepts"]}
    assert "stanley park" in ents                             # from the caption pass
    assert "commuter bike" in ents                            # from the vision pass
    assert "cycling" in concepts                              # from the vision pass
    eng.close()


def test_captioned_image_embed_surface_is_caption_plus_vision():
    # The embedding surface for a captioned image is the caption PLUS the vision description
    # (so the image's visual content is retrievable, not just the caption words). Asserted
    # directly on Ingestor._embed_surface — the vector store indexes exactly this string.
    from kg.corpus import CorpusItem
    from kg.ingest import Ingestor
    d = tempfile.mkdtemp()
    img = _img(d)
    eng = _engine()
    g = eng._g
    ing = Ingestor(g.store, g.extractor, g.embedder, g.canon, g.config)
    item = CorpusItem(id="x", modality="image", source_ref="app/x",
                      text=CAPTION, image_path=img, created_at="2026-06-02T00:00:00Z")
    surface = ing._embed_surface(item, _BIKE_IMAGE_RECORD)
    assert CAPTION in surface and _BIKE_IMAGE_RECORD.description in surface
    # a standalone image's surface stays the bare description
    solo = CorpusItem(id="y", modality="image", source_ref="app/y",
                      text=None, image_path=img, created_at="2026-06-02T00:00:00Z")
    assert ing._embed_surface(solo, _BIKE_IMAGE_RECORD) == _BIKE_IMAGE_RECORD.description
    eng.close()


# --------------------------------------------------------------------------- #
# (c) seamless fit: an image resolves onto a prior conversation's concept nodes
# --------------------------------------------------------------------------- #
def test_image_resolves_onto_existing_conversation_concept_nodes():
    d = tempfile.mkdtemp()
    img = _img(d)
    eng = _engine()
    eng._g.extractor = ScriptedExtractor(
        {CYCLING_NOTE: _CYCLING_NOTE_RECORD, img: _BIKE_IMAGE_RECORD})
    # 1) a plain conversation note first establishes the cycling / commuter-bike nodes
    note = eng.ingest(NoteInput(text=CYCLING_NOTE, created_at="2026-05-01T00:00:00Z"))
    # 2) then an image about the same thing
    image = eng.ingest(NoteInput(text="", attachments=[img],
                                 created_at="2026-06-01T00:00:00Z"))
    store = eng._g.store
    # exactly one shared node per concept/work — the image RESOLVED, it did not fork
    for name in ("cycling", "commuter bike"):
        matches = [n for n in store.nodes_of_type(NodeType.ENTITY)
                   if n.name.lower() == name]
        assert len(matches) == 1, f"{name!r} forked into {len(matches)} nodes"
    # and both episodes actually anchor to that same shared node
    note_ids = set(eng._episode_entity_ids(note.episode_id))
    image_ids = set(eng._episode_entity_ids(image.episode_id))
    assert note_ids & image_ids, "image shares no entity anchor with the prior note"
    eng.close()
