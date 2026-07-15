"""Regression tests for three confirmed ingest/engine fixes:

  1. kg/ingest.py legacy-hash fallback — a store written by the pre-salt build must
     dedup an unchanged re-ingest instead of version-appending ep_X_v1 duplicates.
  2. kg/engine.py media-only notes — empty text is valid when attachments are present,
     and every attachment persists as episode.media_paths.
  3. kg/ingest.py failed extractions — a raising extractor must NOT persist an episode
     or record its hash, so a later retry reprocesses the note from scratch.

All runs use the MOCK provider (no LLM); embeddings use the real local bge model, same
policy as tests/test_engine.py. Run: python -m pytest tests/test_fix_ingest_engine.py -q
"""
from __future__ import annotations

import json
import tempfile

import pytest

from kg.engine import Engine, NoteInput, _MockExtractor
from kg.errors import InvalidInput, ProviderError
from kg.extractors import UsageMeter
from kg.ingest import _sha256
from kg.models import Modality, episode_node


def _open():
    return Engine.open(tempfile.mkdtemp(), {"kind": "mock"})


# --------------------------------------------------------------------------- #
# Finding 1 — legacy-hash dedup for pre-existing (pre-salt) stores
# --------------------------------------------------------------------------- #
def test_legacy_hash_reingest_dedups_and_upgrades():
    eng = _open()
    note = NoteInput(text="Alice met Bob in Paris.", created_at="2024-01-01T00:00:00Z")
    ep_id = eng.ingest(note).episode_id
    store = eng._g.store

    nid = ep_id[len("ep_"):]
    new_h = _sha256("text", note.text, note.created_at, nid)     # current salted formula
    legacy = _sha256("text", note.text)                          # pre-salt formula
    node = store.get_node(ep_id)
    assert node.content_hash == new_h                            # sanity: written salted

    # rewrite the store to look like it was built by the old, unsalted build
    node.content_hash = legacy
    store.hash_cache.pop(new_h, None)
    store.add_hash(legacy, ep_id)

    again = eng.ingest(NoteInput(text=note.text, created_at=note.created_at))
    assert again.skipped                                         # deduped, not re-written
    assert again.episode_id == ep_id
    assert not store.has_node(ep_id + "_v1")                     # no version-append duplicate
    assert eng.episodes_list()["total"] == 1

    # the fallback upgraded the stored hash + cache to the new formula (one-time)
    assert store.get_node(ep_id).content_hash == new_h
    assert new_h in store.hash_cache

    # a third re-ingest now hits the new hash up front and still skips
    third = eng.ingest(NoteInput(text=note.text, created_at=note.created_at))
    assert third.skipped and third.episode_id == ep_id
    assert eng.episodes_list()["total"] == 1
    eng.close()


# --------------------------------------------------------------------------- #
# Finding 2 — media-only notes accepted; attachments persist as media_paths
# --------------------------------------------------------------------------- #
def test_media_only_note_accepted_and_paths_persisted():
    eng = _open()
    note = NoteInput(text="", created_at="2026-07-01T10:00:00Z",
                     attachments=["/tmp/a.png", "/tmp/b.png"])
    res = eng.ingest(note)
    assert not res.skipped
    ep = eng.episode(res.episode_id)
    assert ep is not None
    assert ep["media_paths"] == ["/tmp/a.png", "/tmp/b.png"]     # all attachments reach it
    eng.close()


def test_text_note_with_attachments_keeps_text_and_media():
    eng = _open()
    note = NoteInput(text="Photo from the Berlin trip.", created_at="2026-07-02T10:00:00Z",
                     attachments=["/tmp/berlin.jpg"])
    ep = eng.episode(eng.ingest(note).episode_id)
    assert ep["text"] == "Photo from the Berlin trip."
    assert ep["media_paths"] == ["/tmp/berlin.jpg"]
    eng.close()


def test_readable_attachment_path_is_separate_from_persisted_media_path():
    eng = _open()
    note = NoteInput(text="Photo from the Berlin trip.",
                     created_at="2026-07-02T10:00:00Z",
                     attachments=["/tmp/spool/att01_berlin.jpg"],
                     media_paths=["media/op-berlin.jpg"])
    ep = eng.episode(eng.ingest(note).episode_id)
    assert ep["media_paths"] == ["media/op-berlin.jpg"]
    eng.close()


def test_repair_legacy_media_paths_from_raw_ledger(tmp_path):
    eng = Engine.open(str(tmp_path), {"kind": "mock"})
    text = "Apple Corps image"
    created_at = "2026-07-05T11:15:00+00:00"
    ep_id = "ep_legacy-media"
    eng._g.store.add_node(episode_node(
        ep_id, modality=Modality.TEXT, source_ref="fixture", raw_text=text,
        content_hash="legacy-media", ts=created_at,
    ))
    eng._g.save()
    media = tmp_path / "media"
    media.mkdir()
    (media / "note_0010.png").write_bytes(b"image")
    with open(tmp_path / "raw_inputs.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"created_at": created_at, "text": text,
                            "attachments": ["att01_note_0010.png"]}) + "\n")

    assert eng.repair_legacy_media_paths() == 1
    assert eng.episode(ep_id)["media_paths"] == ["media/note_0010.png"]
    assert eng.repair_legacy_media_paths() == 0
    eng.close()


def test_two_media_only_notes_same_timestamp_do_not_collide():
    eng = _open()
    ts = "2026-07-03T10:00:00Z"
    a = eng.ingest(NoteInput(text="", created_at=ts, attachments=["/tmp/one.png"]))
    b = eng.ingest(NoteInput(text="", created_at=ts, attachments=["/tmp/two.png"]))
    assert a.episode_id != b.episode_id and not a.skipped and not b.skipped
    assert eng.episodes_list()["total"] == 2
    eng.close()


def test_empty_text_without_attachments_still_rejected():
    eng = _open()
    with pytest.raises(InvalidInput):
        eng.ingest(NoteInput(text="   ", created_at="2026-07-01T10:00:00Z"))
    eng.close()


# --------------------------------------------------------------------------- #
# Finding 3 — failed extraction is not persisted; a retry reprocesses it
# --------------------------------------------------------------------------- #
class _BoomExtractor:
    """Extractor whose every extraction raises — stands in for a provider outage."""
    name = "boom"

    def __init__(self):
        self.meter = UsageMeter()

    def extract_text(self, text: str, title: str = ""):
        raise RuntimeError("provider outage")

    def extract_image(self, image_path, label_hint=None):
        raise RuntimeError("provider outage")


def test_failed_extraction_not_committed_and_retryable():
    eng = _open()
    note = NoteInput(text="Carol joined Acme in Denver.", created_at="2026-08-01T09:00:00Z")

    eng._g.extractor = _BoomExtractor()
    with pytest.raises(ProviderError):
        eng.ingest(note)
    # nothing persisted: no episode, no recorded hash → the note is not lost
    assert eng.episodes_list()["total"] == 0
    assert eng._g.store.hash_cache == {}

    # retry with a working extractor reprocesses the SAME note from scratch
    eng._g.extractor = _MockExtractor()
    res = eng.ingest(note)
    assert not res.skipped
    assert res.entities > 0                                       # Carol/Acme/Denver mentions
    assert eng.episodes_list()["total"] == 1
    eng.close()
