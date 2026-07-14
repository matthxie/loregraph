"""Filesystem helpers for the brainbrain app data-dir — the append-only ledger and the
capture spool the daemon consumes.

The app treats the on-disk JSONL ledgers as the source of truth; the SQLite store is a
rebuildable cache. So these helpers are deliberately dumb IO — no dependency on the graph
or engine — and every write is fsynced before returning, because a receipt (or a raw row)
that is not durably on disk is a lost acknowledgment. The data-dir layout mirrored here is
the one the daemon's path resolver defines (kg/_compat/paths.py):

    <data>/raw_inputs.jsonl        append-only raw ledger (sacred, byte-verbatim)
    <data>/extractions/<id>.json   the stored emit_graph payload per note
    <data>/media/                  committed media artifacts
    <data>/ingest/<spool_id>/      the capture spool queue (message.json + copied bytes)
    <data>/ingest/.processed.jsonl the receipt ledger (the ingest acknowledgment)

The message.json and receipt schemas here match the daemon's capture handler and receipt
writer byte-for-byte, because the Discord bot writes the same spool format the daemon and
these helpers read: a drift in either schema silently strands captures.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone

# The spool-id time component: sortable to the microsecond, mirroring the daemon's _TS_FMT.
_TS_FMT = "%Y%m%dT%H%M%S%f"

# Stop chars that terminate a URL scan and the trailing sentence punctuation trimmed off the
# tail — a hand-rolled scan so this module stays inside pure stdlib (no `re`).
_URL_STOP = set(" \t\r\n<>()]")
_URL_TRIM = ".,;:!?'\")"


def _ingest_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "ingest")


def _sanitize_filename(name: str) -> str:
    """Safe spool leaf name, no path escape — basename only,
    no leading dots, every run of unsafe chars collapsed to a single underscore, capped."""
    name = os.path.basename((name or "").replace("\\", "/"))
    name = name.lstrip(".") or "file"
    out: list[str] = []
    prev_us = False
    for ch in name:
        if ch.isascii() and (ch.isalnum() or ch in "._-"):
            out.append(ch)
            prev_us = False
        elif not prev_us:
            out.append("_")
            prev_us = True
    return "".join(out)[:120]


def _extract_urls(text: str) -> list[str]:
    """http(s) URLs, de-duplicated, order-preserving, trailing sentence punctuation trimmed
    — scanned by hand to avoid importing `re`."""
    seen: list[str] = []
    s = text or ""
    low = s.lower()
    n = len(s)
    i = 0
    while True:
        h = low.find("http", i)
        if h == -1:
            break
        if low.startswith("http://", h) or low.startswith("https://", h):
            j = h
            while j < n and s[j] not in _URL_STOP:
                j += 1
            u = s[h:j].rstrip(_URL_TRIM)
            if u and u not in seen:
                seen.append(u)
            i = j
        else:
            i = h + 4
    return seen


def _spool_order_key(ingest_dir: str, name: str) -> tuple[float, int, str]:
    """Authored-time queue order, independent of the spool dir's id format: created_at first,
    mtime as tiebreak/fallback (daemon's spool_order_key)."""
    manifest = os.path.join(ingest_dir, name, "message.json")
    try:
        with open(manifest, encoding="utf-8") as f:
            created_at = json.load(f).get("created_at") or ""
        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp(), os.stat(manifest).st_mtime_ns, name
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        try:
            st = os.stat(manifest)
            return st.st_mtime, st.st_mtime_ns, name
        except OSError:
            return float("inf"), 0, name


def _pending_ids(ingest_dir: str) -> list[str]:
    """Spool dirs (never dotfiles) that carry a manifest, in authored-time order."""
    if not os.path.isdir(ingest_dir):
        return []
    names = [
        name for name in os.listdir(ingest_dir)
        if not name.startswith(".")
        and os.path.isfile(os.path.join(ingest_dir, name, "message.json"))]
    return sorted(names, key=lambda name: _spool_order_key(ingest_dir, name))


def append_raw(data_dir: str, record: dict) -> None:
    """Append one JSON line to <data_dir>/raw_inputs.jsonl (the sacred append-only ledger)."""
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "raw_inputs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def dump_extraction(data_dir: str, note_id: str, payload: dict) -> None:
    """Write <data_dir>/extractions/<note_id>.json atomically (write-tmp, rename) so a reader
    never sees a half-written extraction after a crash mid-write."""
    ext_dir = os.path.join(data_dir, "extractions")
    os.makedirs(ext_dir, exist_ok=True)
    path = os.path.join(ext_dir, f"{note_id}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def spool(data_dir: str, *, text: str = "", media: list[str] | None = None,
          created_at: str | None = None, replaces: str | None = None,
          operation_id: str | None = None,
          operation_fingerprint: str | None = None) -> dict:
    """Stage one capture into <data_dir>/ingest/<spool_id>/message.json, copying any media
    bytes alongside it. Built in a `.tmp-<sid>` dir and renamed into place so a spool only
    ever becomes visible complete (the drainer treats presence of the dir as commit).

    `operation_id` is accepted for signature parity with the daemon's capture RPC — the
    operation-id dedup lives one layer up (it keys off the receipt ledger); at the IO layer
    every call mints a fresh timestamp-ordered spool id.
    """
    media = media or []
    ingest_dir = _ingest_dir(data_dir)
    os.makedirs(ingest_dir, exist_ok=True)

    if created_at:
        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    sid = f"{dt.astimezone(timezone.utc).strftime(_TS_FMT)}-{uuid.uuid4().hex[:8]}"

    staging = os.path.join(ingest_dir, f".tmp-{sid}")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    try:
        usable = [m for m in media if isinstance(m, str) and os.path.isfile(m)]
        manifest = []
        for i, src in enumerate(usable, start=1):
            leaf = f"att{i:02d}_{_sanitize_filename(os.path.basename(src))}"
            shutil.copy2(src, os.path.join(staging, leaf))
            manifest.append({"file": leaf})
        message: dict = {"spool_id": sid, "created_at": dt.isoformat(),
                         "source": "capture"}
        if text:
            message["text"] = text            # byte-verbatim — the raw is sacred
        urls = _extract_urls(text)
        if urls:
            message["urls"] = urls
        if manifest:
            message["attachments"] = manifest
        if replaces:
            message["replaces"] = replaces
        if operation_fingerprint:
            message["operation_fingerprint"] = operation_fingerprint
        with open(os.path.join(staging, "message.json"), "w", encoding="utf-8") as f:
            json.dump(message, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(staging, os.path.join(ingest_dir, sid))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"spool_id": sid, "pending": len(_pending_ids(ingest_dir))}


def list_pending(data_dir: str) -> list[dict]:
    """Pending spools in authored-time order (newest-last): {spool_id, message}."""
    ingest_dir = _ingest_dir(data_dir)
    out: list[dict] = []
    for sid in _pending_ids(ingest_dir):
        try:
            with open(os.path.join(ingest_dir, sid, "message.json"),
                      encoding="utf-8") as f:
                out.append({"spool_id": sid, "message": json.load(f)})
        except (OSError, json.JSONDecodeError):
            continue
    return out


def load_spool(data_dir: str, spool_id: str) -> dict:
    """The spool's message.json dict, augmented with `media`: absolute paths of the copied
    attachment bytes that are actually present (absolute/escaping leafs are dropped as
    hostile — a spool leaf must resolve inside its own dir)."""
    sdir = os.path.join(_ingest_dir(data_dir), spool_id)
    with open(os.path.join(sdir, "message.json"), encoding="utf-8") as f:
        message = json.load(f)
    media: list[str] = []
    for att in message.get("attachments", []):
        leaf = att.get("file") if isinstance(att, dict) else None
        if not leaf or os.path.isabs(leaf):
            continue
        path = os.path.join(sdir, leaf)
        if os.path.isfile(path):
            media.append(path)
    message["media"] = media
    return message


def write_receipt(data_dir: str, spool_id: str, note_ids: list[str], *, at: str,
                  discarded: bool = False, replaces: str | None = None) -> None:
    """Append one receipt line to <data_dir>/ingest/.processed.jsonl (fsynced — the receipt
    IS the acknowledgment) THEN remove the spool dir. `replaces`/`operation_fingerprint` are
    carried from the spool's manifest when not given, so the receipt is self-describing for
    the bot's dedup and edit-tracking."""
    ingest_dir = _ingest_dir(data_dir)
    os.makedirs(ingest_dir, exist_ok=True)
    sdir = os.path.join(ingest_dir, spool_id)

    operation_fingerprint = None
    try:
        with open(os.path.join(sdir, "message.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        replaces = replaces or (manifest.get("replaces") or None)
        operation_fingerprint = manifest.get("operation_fingerprint") or None
    except (OSError, json.JSONDecodeError):
        pass

    row: dict = {"spool_id": spool_id, "note_ids": note_ids, "at": at}
    if discarded:
        row["discarded"] = True
    if replaces:
        row["replaces"] = replaces
    if operation_fingerprint:
        row["operation_fingerprint"] = operation_fingerprint
    with open(os.path.join(ingest_dir, ".processed.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    shutil.rmtree(sdir, ignore_errors=True)


def _last_receipts(data_dir: str, n: int = 20) -> list[dict]:
    """The last `n` receipt rows, newest-first, normalized to the daemon's inbox shape."""
    path = os.path.join(_ingest_dir(data_dir), ".processed.jsonl")
    if not os.path.isfile(path):
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({"spool_id": r.get("spool_id"),
                         "note_ids": r.get("note_ids", []),
                         "at": r.get("at") or r.get("ts"),
                         "discarded": bool(r.get("discarded", False)),
                         "replaces": r.get("replaces") or None,
                         "operation_fingerprint": r.get("operation_fingerprint")})
    return list(reversed(rows[-n:]))


def inbox_status(data_dir: str) -> dict:
    """The inbox snapshot the app polls: pending spools, the (single-process, so always
    False) draining flag, no in-flight batch, and the recent receipts."""
    return {"pending": list_pending(data_dir), "draining": False,
            "batch": None, "last_receipts": _last_receipts(data_dir)}
