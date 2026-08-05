"""Deliberate re-captures revive tombstoned content; bulk re-ingests never do.

The dedup layer skips any item whose content hash is already in the store — including
hashes held by TOMBSTONED episodes, because erasure (kg/forget.py) deliberately keeps
the hash cache so re-ingesting an operator-side copy of an erased session cannot
resurrect the secret. That policy, applied to single deliberate notes, silently ate two
host-app flows: restoring a deleted note from trash (re-capture under the original
created_at → same content hash → skip → nothing restored), and editing a note back to
text it previously held (the replacement dedups onto the tombstone the first edit left).

The fix is a per-item `CorpusItem.revive` flag, set only by Engine.ingest (single-note
captures): a hash/id match on a tombstone falls through to the version-append path and
writes a fresh live `ep_X_vN`, and IngestReport now carries `episode_ids`/`skipped_ids`
so Engine.ingest returns the id that was actually written, not the base-id guess.

All runs use the MOCK provider; embeddings use the real local bge model, same policy as
tests/test_engine.py. Run: python -m pytest tests/test_revive_tombstone.py -q
"""
from __future__ import annotations

import tempfile

from kg.corpus import CorpusItem
from kg.engine import Engine, NoteInput


def _open():
    return Engine.open(tempfile.mkdtemp(), {"kind": "mock"})


def _live_episode_ids(eng) -> list[str]:
    return [e["id"] for e in eng.episodes_list()["episodes"]]


def test_recapture_revives_tombstoned_note():
    eng = _open()
    note = NoteInput(text="Met Sam at the bouldering gym.",
                     created_at="2024-01-01T00:00:00Z")
    first = eng.ingest(note)
    eng.delete_episode(first.episode_id)
    assert _live_episode_ids(eng) == []

    again = eng.ingest(NoteInput(text=note.text, created_at=note.created_at))
    assert not again.skipped, "a deliberate re-capture of deleted content must re-create it"
    assert again.episode_id == f"{first.episode_id}_v1"   # tombstoned ids never reused
    assert _live_episode_ids(eng) == [again.episode_id]
    detail = eng.episode(again.episode_id)
    assert detail is not None and detail["text"] == note.text
    # the dedup cache now answers for the LIVE node, so an identical third capture skips
    # onto it instead of minting _v2
    third = eng.ingest(NoteInput(text=note.text, created_at=note.created_at))
    assert third.skipped and third.episode_id == again.episode_id
    assert _live_episode_ids(eng) == [again.episode_id]
    eng.close()


def test_second_revive_appends_the_next_version():
    eng = _open()
    note = NoteInput(text="Call the dentist about Tuesday.",
                     created_at="2024-02-02T00:00:00Z")
    base = eng.ingest(note).episode_id
    eng.delete_episode(base)
    v1 = eng.ingest(NoteInput(text=note.text, created_at=note.created_at)).episode_id
    assert v1 == f"{base}_v1"
    eng.delete_episode(v1)
    v2 = eng.ingest(NoteInput(text=note.text, created_at=note.created_at)).episode_id
    assert v2 == f"{base}_v2", "each revive mints the next free version id"
    assert _live_episode_ids(eng) == [v2]
    eng.close()


def test_unchanged_reingest_of_a_live_note_still_skips():
    eng = _open()
    note = NoteInput(text="Picked up the keys from Dana.",
                     created_at="2024-03-03T00:00:00Z")
    first = eng.ingest(note)
    again = eng.ingest(NoteInput(text=note.text, created_at=note.created_at))
    assert again.skipped and again.episode_id == first.episode_id
    assert _live_episode_ids(eng) == [first.episode_id]
    eng.close()


def test_bulk_reingest_never_revives_tombstones():
    """The erasure guarantee is untouched: a corpus item (revive defaults False) whose
    content matches a tombstoned episode is still skipped — re-running an import over an
    erased session must not bring the content back."""
    eng = _open()
    item = CorpusItem(id="sess01", modality="text", source_ref="import:test",
                      text="The secret meeting is in the old library.",
                      created_at="2024-04-04T00:00:00Z")
    report = eng._g.ingest([item])
    assert report.episode_ids == ["ep_sess01"]
    eng.delete_episode("ep_sess01")

    again = eng._g.ingest([CorpusItem(id="sess01", modality="text",
                                      source_ref="import:test", text=item.text,
                                      created_at=item.created_at)])
    assert again.skipped == 1 and again.ingested == 0
    assert again.skipped_ids == ["ep_sess01"]             # matched the tombstone, left it dead
    assert not eng._g.store.has_node("ep_sess01_v1")
    assert _live_episode_ids(eng) == []
    eng.close()
