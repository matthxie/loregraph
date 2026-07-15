"""Regression tests for three ledger fixes (pure IO helpers, no daemon/LLM/provider):

  Finding 2 — the raw ledger must gain exactly one row per spool item, even when a drain
              retries a failed item (append_raw_once's durable `.raw` marker).
  Finding 7 — a hostile attachment leaf ('../../etc/passwd', absolute paths) must never let
              out-of-tree bytes into the graph/ledger (contained_leaf / load_spool).
  Finding 9 — a trailing 'Z' UTC suffix must be accepted on Python 3.10 at spool time.

Plus commit_media durability: attachments are copied into <data_dir>/media/ under
stable relative paths, all-or-nothing (a partial copy rolls back).

The daemon-side halves of these regressions (capture/drain RPC wiring) live with the
app-owned daemon in the brainbrain repo, which imports these same helpers.
"""
from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from kg import ledger


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


# --------------------------------------------------------------------- commit_media
def test_commit_media_returns_durable_relative_paths(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.webp"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    paths = ledger.commit_media(str(tmp_path), "op-abc123", [str(first), str(second)])

    assert paths == ["media/op-abc123.png", "media/op-abc123_1.webp"]
    assert (tmp_path / paths[0]).read_bytes() == b"first"
    assert (tmp_path / paths[1]).read_bytes() == b"second"


def test_commit_media_rolls_back_partial_copy(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    original = ledger.shutil.copy2
    copies = 0

    def fail_second(source, destination):
        nonlocal copies
        copies += 1
        if copies == 2:
            raise OSError("injected copy failure")
        return original(source, destination)

    with mock.patch.object(ledger.shutil, "copy2", side_effect=fail_second):
        with pytest.raises(OSError):
            ledger.commit_media(str(tmp_path), "op-rollback", [str(first), str(second)])

    assert not list((tmp_path / "media").glob("op-rollback*"))


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
