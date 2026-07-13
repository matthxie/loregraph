"""kg.daemon — the engine daemon serving engine/PROTOCOL.md over stdio.

    python -m kg.daemon --data <dir>

Newline-delimited JSON-RPC 2.0: requests in on stdin, responses + notifications out on
stdout (stdout carries NOTHING else — logs go to stderr), single-threaded and serial
(PROTOCOL §1). Exit codes: 0 clean · 2 bad argv/data dir · 3 writer lock held.

This is the port that lets the brainbrain Electron app drive THIS package's engine. Every
verb delegates to the two black boxes the app is allowed to see: kg.engine.Engine (the
facade — retrieve/answer/episode(s)/graph_preview/ingest/delete) and kg.ledger (the dumb
fsync-first IO helpers for the capture spool + receipts). The daemon owns only the wire
shaping, the single-writer lock, and the two decoupled side-ledgers the engine has no
opinion about: the tasks list (task_events.jsonl) and the chat history (answers.jsonl).

Protocol coverage: `protocol` is the hard integer gate and stays 1; `protocol_minor` is 1
because the whole v1 core plus the v1.1 additive methods (§7 — episodes.list, episode
lifecycle, chat.history.*, model.ensure, codex.*) are served, but the v1.2 recommender
(§8, recs.*) and the engine-owned rooms are NOT — those answer -32601 METHOD_NOT_FOUND so
the client's capability probe hides their panes. The provider is chosen from the KG_LLM env
via kg.llm_client.current_provider() and handed to Engine.open once, for the process life.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import sys
import threading
import time
from datetime import datetime, timezone

from . import ledger
from .errors import (EngineError, InvalidInput, ModelUnavailable, NotFound,
                     ProviderUnavailable, StoreError)

PROTOCOL_VERSION = 1
# Additive minor (PROTOCOL §7.1): the integer `protocol` is a breaking-change gate and never
# moves; the minor advertises the v1.1 additive surface. 1 ⇒ v1 core + §7 methods are served,
# but §8 recs (which would be minor 2) are not — capability-probing (-32601) is the fallback.
PROTOCOL_MINOR = 1
ENGINE_VERSION = "0.1.0"

# JSON-RPC / PROTOCOL §4 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
NOT_FOUND = -32000
NOT_IMPLEMENTED = -32001
STORE_ERROR = -32002
PROVIDER_UNAVAILABLE = -32003
INVALID_INPUT = -32004
BUSY = -32005
SHUTTING_DOWN = -32006
MODEL_UNAVAILABLE = -32009

_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pid_alive(pid: int) -> bool:
    """Is the lock-holder pid a live process? os.kill(pid, 0) is POSIX-only — on Windows it
    raises, which would otherwise wedge acquire_lock behind any stale lock until hand-deleted."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class RpcError(Exception):
    """A JSON-RPC error the dispatch loop turns into an error envelope with this exact
    code/message/data (as opposed to the EngineError taxonomy, which the loop maps)."""

    def __init__(self, code: int, message: str, data=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class Daemon:
    def __init__(self, data_dir: str, log_level: str = "info"):
        self.data_dir = os.path.abspath(data_dir)
        self.log_level = log_level
        self.started = time.monotonic()
        self.shutting_down = False
        self._engine = None
        self._provider_kind = "openai"
        self._embed_verified = None      # set by model.ensure for health.embedder.verified
        self._last_batch = None          # terminal per-spool states of the most recent drain
        self._draining = False
        # stdout is one JSON value per line (§1); a lock keeps a progress notification from
        # splicing bytes into a response even though v1 dispatch is single-threaded.
        self._send_lock = threading.Lock()

    # ------------------------------------------------------------------ util
    def log(self, level: str, msg: str) -> None:
        order = {"debug": 0, "info": 1, "warn": 2, "error": 3}
        if order.get(level, 1) >= order.get(self.log_level, 1):
            print(f"[{level}] {msg}", file=sys.stderr, flush=True)

    def _send(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._send_lock:
            sys.stdout.write(line)
            sys.stdout.flush()

    def notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    # ------------------------------------------------------------------ engine
    def engine(self):
        """The single long-lived Engine (PROTOCOL §2 single-writer). Provider is read once
        from the KG_LLM env; the embedder inside loads lazily on first vector use, so opening
        this is cheap for health/stats/tasks/capture."""
        if self._engine is None:
            from .engine import Engine
            from .llm_client import current_provider
            provider = current_provider()
            self._provider_kind = (provider.get("kind") or "openai")
            self._engine = Engine.open(self.data_dir, provider=provider, log=self.log)
        return self._engine

    # ------------------------------------------------------------------ lock
    def lock_path(self) -> str:
        return os.path.join(self.data_dir, "daemon.lock")

    def acquire_lock(self) -> bool:
        """O_EXCL lock holding our pid; a stale lock (dead pid) is reclaimed. False → held by
        a live process (the caller exits 3 without emitting ready)."""
        os.makedirs(self.data_dir, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.lock_path(), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return True
            except FileExistsError:
                try:
                    with open(self.lock_path(), encoding="utf-8") as f:
                        pid = int((f.read() or "0").strip() or "0")
                except (OSError, ValueError):
                    pid = 0
                if pid > 0 and _pid_alive(pid):
                    return False
                try:
                    os.unlink(self.lock_path())      # stale — reclaim and retry
                except OSError:
                    return False
        return False

    def release_lock(self) -> None:
        try:
            os.unlink(self.lock_path())
        except OSError:
            pass

    # ------------------------------------------------------------------ params
    @staticmethod
    def _param(params: dict, name: str, *, required: bool = False, default=None,
               types=None):
        if name not in params or params[name] is None:
            if required:
                raise RpcError(INVALID_PARAMS, f"missing required param {name!r}")
            return default
        v = params[name]
        if types and not isinstance(v, types):
            raise RpcError(INVALID_PARAMS, f"param {name!r} has the wrong type")
        return v

    # ------------------------------------------------------------------ wire shaping
    @staticmethod
    def _snippet(text) -> str:
        """EpisodeRef.snippet (PROTOCOL §3): first 200 chars of the raw text, newlines → spaces."""
        return " ".join((text or "").split())[:200]

    @staticmethod
    def _wire_facts(lines) -> list[dict]:
        """The engine hands facts as pre-rendered FactLine strings (kg/rag.py); the wire Fact
        object (§3) is structured, but the only field this v0 engine can honestly fill is
        `rendered`, so each line rides as {"rendered": line}. Structured fact fields are an
        engine followup — see BUILD_REPORT."""
        return [{"rendered": ln} for ln in (lines or []) if isinstance(ln, str)]

    @staticmethod
    def _wire_graph_preview(gp: dict, root_id: str) -> dict:
        """Reshape the engine's {nodes:[{id,name,category}], edges:[{src,dst,label}]} into the
        §3.6/§7.2 display graph: the root is hop 0 kind=episode; its entities are hop 1 with
        their glyph `entity_category`; mention stars are already collapsed to direct
        episode→entity edges. external_connections is 0 (the v0 preview is one hop, so no
        off-screen continuation stubs are inferred)."""
        nodes = []
        for n in gp.get("nodes", []):
            cat = n.get("category")
            is_ep = cat == "episode"
            nodes.append({
                "id": n.get("id"), "label": n.get("name") or "",
                "kind": "episode" if is_ep else "entity",
                "hop": 0 if n.get("id") == root_id else 1,
                "external_connections": 0,
                "entity_category": None if is_ep else cat,
            })
        edges = [{"source": e.get("src"), "target": e.get("dst"),
                  "kind": e.get("label") or "MENTIONS", "label": ""}
                 for e in gp.get("edges", [])]
        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------ read verbs
    def m_health(self, params: dict) -> dict:
        loaded = False
        try:
            loaded = bool(self._engine is not None
                          and getattr(self._engine._g.embedder, "_model", None) is not None)
        except Exception:  # noqa: BLE001 — health never fails on introspection
            loaded = False
        embedder = {"model": _EMBED_MODEL, "loaded": loaded}
        if self._embed_verified is not None:
            embedder["verified"] = self._embed_verified.get("verified", False)
        return {
            "ok": True,
            "engine_version": ENGINE_VERSION,
            "protocol": PROTOCOL_VERSION,
            "protocol_minor": PROTOCOL_MINOR,
            "pid": os.getpid(),
            "uptime_s": round(time.monotonic() - self.started, 1),
            "data_dir": self.data_dir,
            "store_exists": os.path.isfile(
                os.path.join(self.data_dir, "store", "kg.db")),
            "embedder": embedder,
            "llm": {"provider": self._provider_kind},
        }

    def m_stats(self, params: dict) -> dict:
        return self.engine().stats()

    def m_retrieve(self, params: dict) -> dict:
        query = self._param(params, "query", required=True, types=str)
        if not query.strip():
            raise RpcError(INVALID_PARAMS, "param 'query' must be non-empty")
        k = int(self._param(params, "k", default=8, types=(int, float)))
        as_of = self._param(params, "as_of", default=None, types=str)
        res = self.engine().retrieve(query, k=k, as_of=as_of)
        episodes = [{"id": h.get("id"), "title": None,
                     "created_at": h.get("when") or None,
                     "snippet": self._snippet(h.get("text")),
                     "score": h.get("score")}
                    for h in res.get("episodes", [])]
        return {
            "query": res.get("query", query),
            "as_of": res.get("as_of", as_of),
            "episodes": episodes,
            "context_episode_ids": [e["id"] for e in episodes[:6]],
            "facts": self._wire_facts(res.get("facts")),
            "conflicts": [],
            "rendered_text": res.get("rendered_text", ""),
        }

    def m_search(self, params: dict) -> dict:
        # Keyword/BM25 search is an engine facade stub in v0 (kg.engine._STUB); delegating here
        # surfaces its EngineError as -32603 until the engine implements it (BUILD_REPORT).
        terms = self._param(params, "terms", required=True, types=str)
        k = int(self._param(params, "k", default=10, types=(int, float)))
        res = self.engine().search(terms, k=k)
        return {"terms": terms, "episodes": res.get("episodes", []) if isinstance(res, dict)
                else res}

    def m_facts(self, params: dict) -> dict:
        entity = self._param(params, "entity", required=True, types=str)
        as_of = self._param(params, "as_of", default=None, types=str)
        include_closed = bool(self._param(params, "include_closed", default=True, types=bool))
        return self.engine().facts(entity, as_of=as_of, include_closed=include_closed)

    def m_profile(self, params: dict) -> dict:
        as_of = self._param(params, "as_of", default=None, types=str)
        top = int(self._param(params, "top", default=12, types=(int, float)))
        return self.engine().profile(as_of=as_of, top=top)

    def m_episode(self, params: dict) -> dict:
        ep_id = self._param(params, "id", required=True, types=str)
        det = self.engine().episode(ep_id)
        if det is None:
            raise RpcError(NOT_FOUND, f"no episode {ep_id}", {"id": ep_id})
        note_id = ep_id[3:] if ep_id.startswith("ep_") else ep_id
        try:
            gp = self._wire_graph_preview(self.engine().graph_preview(det["id"]), det["id"])
        except EngineError:
            gp = {"nodes": [], "edges": []}
        text = det.get("text") or None
        return {
            "id": det["id"], "note_id": note_id, "title": None,
            "created_at": det.get("created_at") or None,
            "recorded_at": det.get("ingested_at") or None,
            "modality": "text", "source": det.get("source") or "capture",
            "raw_text": text, "description": None, "media_paths": [],
            "concepts": [], "entities": det.get("entities", []),
            "entity_categories": det.get("entity_categories", {}),
            "graph_preview": gp, "facts": [],
        }

    def m_graph_preview(self, params: dict) -> dict:
        node_id = self._param(params, "id", required=True, types=str)
        gp = self.engine().graph_preview(node_id)
        return self._wire_graph_preview(gp, node_id)

    def m_episodes_list(self, params: dict) -> dict:
        offset = self._param(params, "offset", default=0, types=(int, float))
        limit = self._param(params, "limit", default=1000, types=(int, float))
        if isinstance(offset, bool) or isinstance(limit, bool) \
                or int(offset) != offset or int(limit) != limit:
            raise RpcError(INVALID_PARAMS, "'offset'/'limit' must be integers")
        offset = int(offset)
        limit = int(limit)
        if offset < 0:
            raise RpcError(INVALID_PARAMS, "'offset' must be >= 0")
        limit = max(1, min(5000, limit))
        res = self.engine().episodes_list(offset=offset, limit=limit)
        rows = [self._episode_list_row(e) for e in res.get("episodes", [])]
        return {"episodes": rows, "total": res.get("total", len(rows))}

    def _episode_list_row(self, e: dict) -> dict:
        """One §7.2 EpisodeListRow. episodes_list gives only {id, created_at, preview}, so each
        row is enriched from the detail verbs (Engine.episode for raw_text/entities/categories,
        graph_preview for the mini-graph). Best-effort: an engine fault on one row leaves its
        detail fields empty rather than failing the whole list."""
        ep_id = e.get("id")
        row = {"id": ep_id, "title": None, "created_at": e.get("created_at") or None,
               "snippet": self._snippet(e.get("preview")), "raw_text": None,
               "media_paths": [], "modality": "text", "source": "capture",
               "concepts": [], "entities": [], "entity_categories": {},
               "graph_preview": {"nodes": [], "edges": []}}
        try:
            det = self.engine().episode(ep_id)
            if det:
                text = det.get("text") or None
                row["raw_text"] = text
                row["snippet"] = self._snippet(text or e.get("preview"))
                row["source"] = det.get("source") or "capture"
                row["entities"] = det.get("entities", [])
                row["entity_categories"] = det.get("entity_categories", {})
            row["graph_preview"] = self._wire_graph_preview(
                self.engine().graph_preview(ep_id), ep_id)
        except EngineError:
            pass
        return row

    # ------------------------------------------------------------------ chat.answer
    def m_chat_answer(self, params: dict) -> dict:
        question = self._param(params, "question", required=True, types=str)
        if not question.strip():
            raise RpcError(INVALID_PARAMS, "param 'question' must be non-empty")
        k = int(self._param(params, "k", default=8, types=(int, float)))
        as_of = self._param(params, "as_of", default=None, types=str)
        try:
            ans = self.engine().answer(question, k=k, as_of=as_of)
        except ProviderUnavailable as e:
            # The whole-request degradation path (§3.12): the client falls back to `retrieve`
            # and renders the context block itself.
            raise RpcError(PROVIDER_UNAVAILABLE, f"provider_unavailable: {e}",
                           {"fallback": "retrieve"})
        ctx = ans.get("context", {})
        # The citation gate lives in the engine's rag layer (kg/rag.py): cited ids not in the
        # retrieved context are already dropped into invalid_citations. Surface both verbatim.
        return {
            "answer": ans.get("answer", ""),
            "citations": ans.get("citations", []),
            "invalid_citations": ans.get("invalid_citations", []),
            "context": {
                "episode_ids": ctx.get("episodes", []),
                "facts": self._wire_facts(ctx.get("facts")),
                "conflicts": [],
                "rendered_text": ctx.get("rendered_text", ""),
            },
        }

    # ------------------------------------------------------------------ capture / inbox
    def m_capture(self, params: dict) -> dict:
        text = self._param(params, "text", default="", types=str) or ""
        media = self._param(params, "media", default=[], types=list) or []
        created_at = self._param(params, "created_at", default=None, types=str)
        replaces = self._param(params, "replaces", default=None, types=str)
        usable = [m for m in media if isinstance(m, str) and os.path.isfile(m)]
        if not text.strip() and not usable:
            # Not an error envelope (§3.9): the client treats "empty" as a validation failure.
            return {"status": "empty", "spool_id": None,
                    "pending": len(ledger.list_pending(self.data_dir)), "episode_id": None}
        if created_at:
            try:
                datetime.fromisoformat(created_at)
            except ValueError:
                raise RpcError(INVALID_INPUT, f"bad ISO date for 'created_at': {created_at!r}")
        try:
            res = ledger.spool(self.data_dir, text=text, media=usable,
                               created_at=created_at, replaces=replaces)
        except OSError as e:
            raise StoreError(f"capture could not spool: {e}")
        return {"status": "spooled", "spool_id": res["spool_id"],
                "pending": res["pending"], "episode_id": None}

    def m_inbox_status(self, params: dict) -> dict:
        snap = ledger.inbox_status(self.data_dir)
        pending = [self._wire_pending(p["spool_id"], p["message"])
                   for p in snap.get("pending", [])]
        return {"pending": pending, "draining": self._draining,
                "batch": self._last_batch, "last_receipts": snap.get("last_receipts", [])}

    def _wire_pending(self, sid: str, message: dict) -> dict:
        """§3.10 pending row: the manifest fields plus the attachment leaf names whose bytes are
        actually present in the spool dir (a spool leaf must resolve inside its own dir; an
        absolute path in a manifest is hostile and dropped)."""
        sdir = os.path.join(self.data_dir, "ingest", sid)
        atts = []
        for a in message.get("attachments", []):
            leaf = a.get("file") if isinstance(a, dict) else None
            if leaf and not os.path.isabs(leaf) and os.path.isfile(os.path.join(sdir, leaf)):
                atts.append(leaf)
        return {"spool_id": message.get("spool_id", sid),
                "created_at": message.get("created_at") or None,
                "source": message.get("source") or "discord",
                "text": message.get("text", ""), "urls": message.get("urls", []),
                "attachments": atts, "replaces": message.get("replaces") or None}

    def m_inbox_drain(self, params: dict) -> dict:
        """Extract + ingest every pending spool (or a chosen subset), returning receipts (§3.11).
        Per item, in order: settled guard → load spool → append the sacred raw row → engine
        ingest → dump the extraction → receipt-then-delete. Each item's failure fails that item
        only (it stays spooled for the next drain, no receipt); an `inbox.progress` notification
        rides every item. v1 dispatch is serial so a concurrent second drain cannot arrive; the
        BUSY guard is kept only for defence in depth."""
        if self._draining:
            raise RpcError(BUSY, "a drain is already running")
        from .engine import NoteInput
        want = self._param(params, "spool_ids", default=None, types=list)
        wanted = set(want) if want is not None else None
        pending = [p for p in ledger.list_pending(self.data_dir)
                   if wanted is None or p["spool_id"] in wanted]
        total = len(pending)
        receipts: list[dict] = []
        ok = failed = skipped = 0
        self._draining = True
        try:
            for i, item in enumerate(pending):
                sid = item["spool_id"]
                try:
                    spool = ledger.load_spool(self.data_dir, sid)
                except (OSError, ValueError, json.JSONDecodeError):
                    skipped += 1
                    receipts.append({"spool_id": sid, "status": "skipped", "at": _now_iso()})
                    self.notify("inbox.progress", {"spool_id": sid, "state": "skipped",
                                                   "done": ok, "total": total})
                    continue
                self.notify("inbox.progress", {"spool_id": sid, "state": "processing",
                                               "done": ok, "total": total})
                try:
                    receipt = self._drain_one(NoteInput, sid, spool)
                except Exception as e:  # noqa: BLE001 — per-item failure, batch continues
                    failed += 1
                    at = _now_iso()
                    receipts.append({"spool_id": sid, "status": "failed",
                                     "error": str(e) or "ingest failed", "at": at})
                    self.notify("inbox.progress", {"spool_id": sid, "state": "failed",
                                                   "error": str(e) or "ingest failed",
                                                   "done": ok, "total": total})
                    continue
                ok += 1
                receipts.append(receipt)
                self.notify("inbox.progress", {"spool_id": sid, "state": "done",
                                               "episode_id": receipt["episode_id"],
                                               "done": ok, "total": total})
        finally:
            self._draining = False
        self._last_batch = {
            "total": len(receipts), "current": None,
            "done": sum(1 for r in receipts if r["status"] == "ingested"),
            "items": {r["spool_id"]: ({"state": "done", "episode_id": r.get("episode_id")}
                                      if r["status"] == "ingested"
                                      else {"state": r["status"], "error": r.get("error")})
                      for r in receipts}}
        return {"receipts": receipts, "ok": ok, "failed": failed, "skipped": skipped,
                "pending_remaining": len(ledger.list_pending(self.data_dir))}

    def _drain_one(self, NoteInput, sid: str, spool: dict) -> dict:
        """One spool → one episode: append the raw row, ingest through the engine (content-hash
        idempotent, so a retried raw dedups on rebuild), dump the extraction summary, then the
        receipt (which also removes the spool dir). Raises on any leg to fail just this item."""
        text = spool.get("text", "") or ""
        created_at = spool.get("created_at") or _now_iso()
        source = spool.get("source") or "capture"
        media = spool.get("media", [])
        replaces = spool.get("replaces") or None
        ledger.append_raw(self.data_dir, {
            "spool_id": sid, "created_at": created_at, "source": source, "text": text,
            "urls": spool.get("urls", []),
            "attachments": [os.path.basename(m) for m in media]})
        res = self.engine().ingest(NoteInput(text=text, created_at=created_at,
                                             attachments=media, source=source))
        note_id = res.episode_id[3:] if res.episode_id.startswith("ep_") else res.episode_id
        ledger.dump_extraction(self.data_dir, note_id, {
            "episode_id": res.episode_id, "entities": res.entities,
            "relations": res.relations, "concepts": res.concepts, "skipped": res.skipped})
        at = _now_iso()
        ledger.write_receipt(self.data_dir, sid, [res.episode_id], at=at, replaces=replaces)
        return {"spool_id": sid, "status": "ingested", "episode_id": res.episode_id, "at": at}

    # ------------------------------------------------------------------ note lifecycle
    def m_episode_delete(self, params: dict) -> dict:
        ep_id = self._param(params, "id", required=True, types=str)
        if self.engine().episode(ep_id) is None:      # unknown OR already-tombstoned
            return {"ok": True, "id": ep_id, "already_deleted": True}
        self.engine().delete_episode(ep_id)
        return {"ok": True, "id": ep_id}

    def m_episode_reingest(self, params: dict) -> dict:
        """Edit & reprocess (§7.5): spool the edited text as a fresh capture keeping the OLD
        created_at + surviving media (replaces=old id), then delete the old note so a new id is
        minted on the next drain. The whole span runs in one serial request, ahead of any queued
        inbox.drain, so nothing observes the half-swapped state."""
        ep_id = self._param(params, "id", required=True, types=str)
        text = self._param(params, "text", default="", types=str) or ""
        media = self._param(params, "media", default=[], types=list) or []
        res = self.engine().episode(ep_id)
        if res is None:
            raise RpcError(NOT_FOUND, f"no episode {ep_id}", {"id": ep_id})
        surviving = [m for m in media if isinstance(m, str) and os.path.isfile(m)]
        if not text.strip() and not surviving:
            raise RpcError(INVALID_INPUT,
                           "reingest needs non-empty text or at least one surviving media file")
        try:
            cap = ledger.spool(self.data_dir, text=text, media=surviving,
                               created_at=res.get("created_at"), replaces=res["id"])
        except OSError as e:
            raise StoreError(f"reingest could not spool the new capture: {e}")
        self.engine().delete_episode(res["id"])
        return {"ok": True, "spool_id": cap["spool_id"],
                "pending": len(ledger.list_pending(self.data_dir))}

    # ------------------------------------------------------------------ tasks (decoupled)
    def _tasks_path(self) -> str:
        return os.path.join(self.data_dir, "task_events.jsonl")

    def _load_task_rows(self) -> dict:
        """Replay task_events.jsonl into {id: row}. The ledger — not any table — is the durable
        truth (§6.3); the op vocabulary is add|done|reopen|dismiss|edit, wider than the v1 method
        surface, so future methods need no protocol bump."""
        path = self._tasks_path()
        rows: dict = {}
        if not os.path.isfile(path):
            return rows
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                op = ev.get("op")
                tid = ev.get("task")
                if not tid:
                    continue
                if op == "add":
                    rows[tid] = {"id": tid, "title": ev.get("title", ""),
                                 "due": ev.get("due") or None, "status": "open",
                                 "created_at": ev.get("at"), "completed_at": None}
                elif tid not in rows:
                    continue
                elif op == "done":
                    rows[tid]["status"] = "done"
                    rows[tid]["completed_at"] = ev.get("at")
                elif op == "dismiss":
                    rows[tid]["status"] = "dismissed"
                    rows[tid]["completed_at"] = ev.get("at")
                elif op == "reopen":
                    rows[tid]["status"] = "open"
                    rows[tid]["completed_at"] = None
                elif op == "edit":
                    if "title" in ev:
                        rows[tid]["title"] = ev["title"]
                    if "due" in ev:
                        rows[tid]["due"] = ev["due"] or None
        return rows

    def _append_task_event(self, ev: dict) -> None:
        ev = {**ev, "at": ev.get("at") or _now_iso()}
        path = self._tasks_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:      # fsync before ack (§2)
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def m_tasks_list(self, params: dict) -> dict:
        status = self._param(params, "status", default=None, types=str)
        if status is not None and status not in ("open", "done", "dismissed"):
            raise RpcError(INVALID_PARAMS, "param 'status' must be open|done|dismissed")
        rows = [r for r in self._load_task_rows().values()
                if status is None or r["status"] == status]
        # §3.8 order: open first, then created_at descending (stable two-pass).
        rows.sort(key=lambda t: t["created_at"] or "", reverse=True)
        rows.sort(key=lambda t: 0 if t["status"] == "open" else 1)
        return {"tasks": rows}

    def m_tasks_add(self, params: dict) -> dict:
        title = (self._param(params, "title", required=True, types=str) or "").strip()
        if not title:
            raise RpcError(INVALID_INPUT, "empty task title")
        due = self._param(params, "due", default=None, types=str)
        if due is not None and not _is_iso_date(due):
            raise RpcError(INVALID_INPUT, f"bad ISO date for 'due': {due!r}")
        tid = "task_" + secrets.token_hex(3)
        ev = {"op": "add", "task": tid, "title": title}
        if due:
            ev["due"] = due
        self._append_task_event(ev)
        return {"task": self._load_task_rows()[tid]}

    def m_tasks_close(self, params: dict) -> dict:
        tid = self._param(params, "id", required=True, types=str)
        rows = self._load_task_rows()
        if tid not in rows:
            raise RpcError(NOT_FOUND, f"no task {tid}", {"id": tid})
        if rows[tid]["status"] != "done":       # closing an already-done task is a no-op
            self._append_task_event({"op": "done", "task": tid})
            rows = self._load_task_rows()
        open_remaining = sum(1 for r in rows.values() if r["status"] == "open")
        return {"task": rows[tid], "open_remaining": open_remaining}

    # ------------------------------------------------------------------ chat history
    def _answers_path(self) -> str:
        return os.path.join(self.data_dir, "answers.jsonl")

    def m_chat_history_append(self, params: dict) -> dict:
        turn = self._param(params, "turn", required=True, types=dict)
        if not isinstance(turn.get("question"), str) or not isinstance(turn.get("answer"), str):
            raise RpcError(INVALID_INPUT, "turn.question and turn.answer must be strings")
        path = self._answers_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:      # fsync before ack (§2)
            f.write(json.dumps(turn, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return {"ok": True}

    def m_chat_history_list(self, params: dict) -> dict:
        conv = self._param(params, "conversation_id", default=None, types=str)
        path = self._answers_path()
        turns: list[dict] = []
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        turn = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if conv is None or turn.get("conversation_id") == conv:
                        turns.append(turn)
        return {"turns": turns}

    # ------------------------------------------------------------------ model / codex
    def m_model_ensure(self, params: dict) -> dict:
        """Eager, checksum-verified model provisioning (§7.7). kg.modelpin does the work
        (pinned artifact, determinate progress by cache-dir polling, wipe-on-mismatch);
        a hard download failure raises ModelUnavailable → -32009 via the dispatcher."""
        from . import modelpin
        result = modelpin.ensure(
            notify=lambda p: self.notify("model.progress", p), log=self.log)
        # Record verify state for the additive health.embedder.verified field (§7.7).
        self._embed_verified = {"verified": result["state"] == "ready",
                                "sha256": result.get("sha256")}
        return result

    def m_codex_usage(self, params: dict) -> dict:
        from .llm_client import provider_usage
        return provider_usage("codex")            # fails soft, never an error envelope

    def m_codex_signout(self, params: dict) -> dict:
        from .llm_client import provider_signout
        return provider_signout("codex")

    def m_shutdown(self, params: dict) -> dict:
        self.shutting_down = True
        return {"ok": True}

    # ------------------------------------------------------------------ dispatch
    METHODS = {
        "health": m_health,
        "stats": m_stats,
        "retrieve": m_retrieve,
        "search": m_search,
        "facts": m_facts,
        "profile": m_profile,
        "episode": m_episode,
        "graph.preview": m_graph_preview,
        "episodes.list": m_episodes_list,
        "chat.answer": m_chat_answer,
        "capture": m_capture,
        "inbox.status": m_inbox_status,
        "inbox.drain": m_inbox_drain,
        "episode.delete": m_episode_delete,
        "episode.reingest": m_episode_reingest,
        "tasks.list": m_tasks_list,
        "tasks.add": m_tasks_add,
        "tasks.close": m_tasks_close,
        "chat.history.append": m_chat_history_append,
        "chat.history.list": m_chat_history_list,
        "model.ensure": m_model_ensure,
        "codex.usage": m_codex_usage,
        "codex.signout": m_codex_signout,
        "shutdown": m_shutdown,
        # NOT served (respond -32601 via the unknown-method path so the client capability probe
        # hides their panes): recs.list/refresh/feedback/signal/archive/cancel (§8, minor 2) and
        # rooms.list/rooms.episodes.
    }

    def _handle_line(self, line: str) -> None:
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            self._send({"jsonrpc": "2.0", "id": None,
                        "error": {"code": PARSE_ERROR, "message": "parse error"}})
            return
        if not isinstance(req, dict) or req.get("jsonrpc") != "2.0" \
                or not isinstance(req.get("method"), str):
            self._send({"jsonrpc": "2.0",
                        "id": req.get("id") if isinstance(req, dict) else None,
                        "error": {"code": INVALID_REQUEST, "message": "invalid request"}})
            return
        rid = req.get("id")
        is_notification = "id" not in req
        method = req["method"]
        params = req.get("params") or {}
        if not isinstance(params, dict):
            if not is_notification:
                self._send({"jsonrpc": "2.0", "id": rid,
                            "error": {"code": INVALID_PARAMS,
                                      "message": "params must be an object"}})
            return

        def reply(payload: dict) -> None:
            if not is_notification:
                self._send({"jsonrpc": "2.0", "id": rid, **payload})

        if self.shutting_down:
            reply({"error": {"code": SHUTTING_DOWN, "message": "shutting down"}})
            return
        fn = self.METHODS.get(method)
        if fn is None:
            reply({"error": {"code": METHOD_NOT_FOUND, "message": f"no method {method!r}"}})
            return
        try:
            reply({"result": fn(self, params)})
        except RpcError as e:
            err = {"code": e.code, "message": e.message}
            if e.data is not None:
                err["data"] = e.data
            reply({"error": err})
        except InvalidInput as e:
            reply({"error": {"code": INVALID_PARAMS, "message": str(e)}})
        except NotFound as e:
            reply({"error": {"code": NOT_FOUND, "message": str(e)}})
        except ProviderUnavailable as e:
            reply({"error": {"code": PROVIDER_UNAVAILABLE, "message": str(e)}})
        except StoreError as e:
            reply({"error": {"code": STORE_ERROR, "message": str(e)}})
        except ModelUnavailable as e:
            reply({"error": {"code": MODEL_UNAVAILABLE, "message": str(e)}})
        except EngineError as e:
            reply({"error": {"code": INTERNAL_ERROR, "message": str(e) or "engine error"}})
        except Exception as e:  # noqa: BLE001 — uncaught → INTERNAL_ERROR, full trace to stderr
            import traceback
            tb = traceback.format_exc()
            self.log("error", f"{method} raised: {e}\n{tb}")
            reply({"error": {"code": INTERNAL_ERROR, "message": str(e) or "internal error",
                             "data": {"trace": tb[-2000:]}}})

    def run(self) -> int:
        if not self.acquire_lock():
            print(f"daemon.lock held by a live process under {self.data_dir}; refusing "
                  f"to double-write (exit 3)", file=sys.stderr, flush=True)
            return 3

        def _sigterm(_sig, _frm):
            if self.shutting_down:
                return
            self.shutting_down = True
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _sigterm)
        try:
            # §2: open (or create) the store, THEN handshake — the embedder loads lazily so
            # ready stays fast even on first run.
            self.engine()
            self.notify("ready", {"protocol": PROTOCOL_VERSION,
                                  "protocol_minor": PROTOCOL_MINOR,
                                  "engine_version": ENGINE_VERSION,
                                  "pid": os.getpid(), "data_dir": self.data_dir})
            self.log("info", f"ready — data dir {self.data_dir}")
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                self._handle_line(line)
                if self.shutting_down:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            # The EOF path arrives here with shutting_down still False; flip it and disarm the
            # handler so a supervisor SIGTERM racing our exit can't raise mid-teardown.
            self.shutting_down = True
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                if self._engine is not None:
                    self._engine.close()
            except Exception as e:  # noqa: BLE001
                self.log("error", f"final save failed: {e}")
            self.release_lock()
            self.log("info", "shut down cleanly")
        return 0


def _is_iso_date(s: str) -> bool:
    """A bare ISO date/year the `due` field accepts: YYYY, YYYY-MM, or YYYY-MM-DD."""
    parts = s.split("-")
    if not 1 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        return False
    widths = (4, 2, 2)
    return all(len(p) == w for p, w in zip(parts, widths))


def main(argv=None) -> int:
    if os.name == "nt":
        # §1 mandates UTF-8 framing; Windows pipes default to the ANSI code page, so the first
        # non-ASCII byte in a response would kill the daemon without this.
        for stream in (sys.stdin, sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m kg.daemon", add_help=True)
    p.add_argument("--data", default=None)
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warn", "error"])
    args = p.parse_args(argv)
    if not args.data:
        print("kg.daemon: --data <dir> is required (no ./graph-data fallback here)",
              file=sys.stderr, flush=True)
        return 2
    data = os.path.abspath(args.data)
    parent = os.path.dirname(data)
    if parent and not os.path.isdir(parent):
        print(f"kg.daemon: data dir parent does not exist: {parent}",
              file=sys.stderr, flush=True)
        return 2
    return Daemon(data, args.log_level).run()


if __name__ == "__main__":
    sys.exit(main())
