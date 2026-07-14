"""Regression tests for three daemon/ledger fixes:

  Finding 2 — the raw ledger must gain exactly one row per spool item, even when a drain
              retries a failed item (the spool stays queued on failure).
  Finding 7 — a hostile attachment leaf ('../../etc/passwd', absolute paths) must never let
              out-of-tree bytes into the graph/ledger, at both spool-read sites.
  Finding 9 — a trailing 'Z' UTC suffix must be accepted on Python 3.10 at capture and spool.

These exercise the pure IO helpers directly plus the daemon wiring with a stubbed engine, so
no LLM/provider is needed.
"""
from __future__ import annotations

import json
import os

import pytest

from kg import ledger
from kg.daemon import INVALID_INPUT, Daemon, RpcError


# --------------------------------------------------------------------------- helpers
class _FakeNote:
    """Stand-in for engine.NoteInput (owned by a concurrent agent); _drain_one just calls it."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Res:
    def __init__(self, ep_id="ep_note1"):
        self.episode_id = ep_id
        self.entities = 0
        self.relations = 0
        self.concepts = 0
        self.skipped = False


class _FlakyEngine:
    """Fails the first ingest, succeeds after — models a failed drain that re-drains."""
    def __init__(self, fail_times=1):
        self.calls = 0
        self.fail_times = fail_times

    def ingest(self, note):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("boom")
        return _Res()


def _daemon(tmp_path, engine=None):
    d = Daemon(str(tmp_path))
    d.notify = lambda *a, **k: None          # keep progress notifications off stdout
    if engine is not None:
        d.engine = lambda: engine
    return d


def _raw_rows(data_dir):
    path = os.path.join(data_dir, "raw_inputs.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- Finding 9
def test_normalize_iso_maps_z_to_offset():
    assert ledger.normalize_iso("2026-07-14T00:00:00Z") == "2026-07-14T00:00:00+00:00"
    assert ledger.normalize_iso("2026-07-14T00:00:00z") == "2026-07-14T00:00:00+00:00"
    # non-Z inputs are untouched
    assert ledger.normalize_iso("2026-07-14T00:00:00+00:00") == "2026-07-14T00:00:00+00:00"
    assert ledger.normalize_iso("") == ""


def test_spool_accepts_z_created_at(tmp_path):
    # Previously raised ValueError on Python 3.10; must spool cleanly now.
    res = ledger.spool(str(tmp_path), text="hi", created_at="2026-07-14T00:00:00Z")
    msg = ledger.load_spool(str(tmp_path), res["spool_id"])
    # created_at is normalized to an offset the manifest can round-trip.
    assert msg["created_at"].endswith("+00:00")


def test_m_capture_accepts_z_created_at(tmp_path):
    d = _daemon(tmp_path)
    out = d.m_capture({"text": "hello", "created_at": "2026-07-14T00:00:00Z"})
    assert out["status"] == "spooled"


def test_m_capture_rejects_bad_created_at(tmp_path):
    d = _daemon(tmp_path)
    with pytest.raises(RpcError) as ei:
        d.m_capture({"text": "hello", "created_at": "not-a-date"})
    assert ei.value.code == INVALID_INPUT


# --------------------------------------------------------------------------- Finding 7
def test_contained_leaf_rejects_traversal_and_absolute(tmp_path):
    base = str(tmp_path)
    assert ledger.contained_leaf(base, "../../etc/passwd") is None
    assert ledger.contained_leaf(base, "/etc/passwd") is None
    assert ledger.contained_leaf(base, "") is None
    assert ledger.contained_leaf(base, None) is None
    # a plain contained leaf resolves inside base
    good = ledger.contained_leaf(base, "att01_file.bin")
    assert good == os.path.join(os.path.realpath(base), "att01_file.bin")


def _write_manifest(sdir, attachments):
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, "message.json"), "w", encoding="utf-8") as f:
        json.dump({"spool_id": os.path.basename(sdir), "created_at": "2026-07-14T00:00:00+00:00",
                   "source": "capture", "text": "t", "attachments": attachments}, f)


def test_load_spool_drops_escaping_leaf(tmp_path):
    # An out-of-tree secret the hostile manifest tries to reach.
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    sid = "20260714T000000000000-deadbeef"
    sdir = os.path.join(str(tmp_path), "ingest", sid)
    # legit contained attachment
    _write_manifest(sdir, [{"file": "att01_ok.bin"}, {"file": "../../secret.txt"}])
    with open(os.path.join(sdir, "att01_ok.bin"), "w", encoding="utf-8") as f:
        f.write("ok")

    msg = ledger.load_spool(str(tmp_path), sid)
    # only the contained attachment survives; the traversal leaf is dropped
    assert msg["media"] == [os.path.join(sdir, "att01_ok.bin")]


def test_wire_pending_drops_escaping_leaf(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    sid = "20260714T000000000000-cafef00d"
    sdir = os.path.join(str(tmp_path), "ingest", sid)
    _write_manifest(sdir, [{"file": "att01_ok.bin"}, {"file": "../../secret.txt"},
                           {"file": "/etc/passwd"}])
    with open(os.path.join(sdir, "att01_ok.bin"), "w", encoding="utf-8") as f:
        f.write("ok")

    d = _daemon(tmp_path)
    with open(os.path.join(sdir, "message.json"), encoding="utf-8") as f:
        message = json.load(f)
    row = d._wire_pending(sid, message)
    assert row["attachments"] == ["att01_ok.bin"]


# --------------------------------------------------------------------------- Finding 2
def test_append_raw_once_is_idempotent_per_spool(tmp_path):
    sid = "20260714T000000000000-11112222"
    sdir = os.path.join(str(tmp_path), "ingest", sid)
    os.makedirs(sdir)
    rec = {"spool_id": sid, "text": "x"}
    assert ledger.append_raw_once(str(tmp_path), sid, rec) is True
    assert ledger.append_raw_once(str(tmp_path), sid, rec) is False
    rows = _raw_rows(str(tmp_path))
    assert len(rows) == 1 and rows[0]["spool_id"] == sid


def test_drain_retry_does_not_duplicate_raw_row(tmp_path):
    # Spool one item, then drive _drain_one twice: first attempt fails after the raw append,
    # the item stays queued, the second attempt succeeds. The raw ledger must hold ONE row.
    res = ledger.spool(str(tmp_path), text="hello world")
    sid = res["spool_id"]
    engine = _FlakyEngine(fail_times=1)
    d = _daemon(tmp_path, engine=engine)

    spool = ledger.load_spool(str(tmp_path), sid)
    with pytest.raises(RuntimeError):
        d._drain_one(_FakeNote, sid, spool)          # fails inside engine.ingest

    # spool still present (no receipt), so it re-drains
    assert any(p["spool_id"] == sid for p in ledger.list_pending(str(tmp_path)))
    spool = ledger.load_spool(str(tmp_path), sid)
    receipt = d._drain_one(_FakeNote, sid, spool)    # succeeds this time
    assert receipt["status"] == "ingested"

    rows = _raw_rows(str(tmp_path))
    assert [r["spool_id"] for r in rows] == [sid]     # exactly one raw row


def test_full_drain_retry_via_m_inbox_drain(tmp_path):
    # End-to-end through the public drain method: first drain fails the item, second succeeds;
    # raw ledger still holds a single row for the spool.
    res = ledger.spool(str(tmp_path), text="note text")
    sid = res["spool_id"]
    engine = _FlakyEngine(fail_times=1)
    d = _daemon(tmp_path, engine=engine)

    out1 = d.m_inbox_drain({})
    assert out1["failed"] == 1 and out1["ok"] == 0
    out2 = d.m_inbox_drain({})
    assert out2["ok"] == 1 and out2["failed"] == 0

    rows = _raw_rows(str(tmp_path))
    assert [r["spool_id"] for r in rows] == [sid]
