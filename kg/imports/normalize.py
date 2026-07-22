"""Canonical Conversations → ingestible CorpusItems (segmentation + inline images).

This is the source-agnostic funnel: once a mapper has produced `Conversation`s, everything
here is identical regardless of which app the export came from.

Session segmentation
    One Conversation → one or more sessions. We prefer the source's conversation id and split
    only on a large time gap between consecutive messages (a multi-day thread is really
    several sittings). Each session becomes ONE text-episode CorpusItem dated at the session
    start, so the bi-temporal layer orders it by real chat time.

Session text format
    Byte-compatible with what the engine consumes (scripts/build_longmemeval.render_session)
    so the "turns" chunker (kg/chunkers.py) splits it: a `[chat session — YYYY/MM/DD (Day)
    HH:MM]` header, then `User: …` / `Assistant: …` turn lines.

Inline images (perceive-and-inline)
    For each image with resolvable bytes we call the extractor's vision pass and splice its
    one-line description into the turn as `[image: <description>]`; absent bytes yield
    `[image: unavailable]`. This keeps ONE episode per session (no new episode plumbing) while
    still making the image content searchable through the transcript.
"""
from __future__ import annotations

import os
import tempfile

from ..corpus import CorpusItem
from .canonical import Conversation, Media, Message
from .timeutil import header_date, to_datetime

# A gap larger than this between two consecutive messages ends a session (the thread was
# picked up in a separate sitting). 4h balances "same conversation, slept on it" against
# "genuinely a new session".
SESSION_GAP_SECONDS = 4 * 3600

_ROLE_LABEL = {"user": "User", "assistant": "Assistant", "system": "System", "tool": "Tool"}


class NormalizeStats:
    def __init__(self) -> None:
        self.sessions = 0
        self.images_perceived = 0
        self.images_unavailable = 0


def to_corpus_items(conversations: list[Conversation], extractor,
                    stats: NormalizeStats | None = None) -> list[CorpusItem]:
    """Flatten Conversations into session CorpusItems, perceiving inline images via
    `extractor.extract_image`. `stats` (optional) accumulates session/image counts."""
    stats = stats or NormalizeStats()
    items: list[CorpusItem] = []
    for conv in conversations:
        sessions = _segment(conv)
        multi = len(sessions) > 1
        for i, msgs in enumerate(sessions):
            text = _render_session(msgs, conv, stats, extractor)
            if not text:
                continue
            sid = f"{conv.id}#s{i:02d}" if multi else conv.id
            created = _session_start(msgs) or conv.created_at
            items.append(CorpusItem(
                id=sid, modality="text",
                source_ref=f"import:{conv.id}", title=conv.title or "",
                text=text, created_at=created))
            stats.sessions += 1
    return items


def _segment(conv: Conversation) -> list[list[Message]]:
    """Split a conversation's messages into sessions on large inter-message time gaps.
    Messages with no time inherit the running clock, so a partially-timed thread still
    segments sensibly instead of exploding into singletons."""
    if not conv.messages:
        return []
    sessions: list[list[Message]] = [[]]
    prev = None
    for msg in conv.messages:
        dt = to_datetime(msg.created_at)
        if prev is not None and dt is not None \
                and (dt - prev).total_seconds() > SESSION_GAP_SECONDS:
            sessions.append([])
        sessions[-1].append(msg)
        if dt is not None:
            prev = dt
    return [s for s in sessions if s]


def _session_start(msgs: list[Message]) -> str | None:
    for m in msgs:
        if m.created_at:
            return m.created_at
    return None


def _render_session(msgs: list[Message], conv: Conversation,
                    stats: NormalizeStats, extractor) -> str:
    """One session → the `[chat session — …]` transcript, images perceived inline."""
    start = _session_start(msgs) or conv.created_at
    dt = to_datetime(start)
    header = f"[chat session — {header_date(dt)}]" if dt else "[chat session]"
    lines = [header]
    for msg in msgs:
        who = _ROLE_LABEL.get(msg.role, "User")
        body = msg.text.strip()
        for media in msg.media:
            marker = _perceive(media, extractor, stats)
            if marker:
                body = f"{body}\n{marker}" if body else marker
        if not body:
            continue
        lines.append(f"{who}: {body}")
    # a header with no turns is not a real session
    return "\n".join(lines) if len(lines) > 1 else ""


def _perceive(media: Media, extractor, stats: NormalizeStats) -> str | None:
    """Perceive one image → `[image: <one-line description>]`, or `[image: unavailable]`
    when its bytes are not in the export. Never raises — a vision failure degrades to the
    alt/placeholder so one bad attachment can't sink an import."""
    if media.kind != "image":
        return None
    path, tmp = media.path, None
    try:
        if (not path or not os.path.isfile(path)) and media.data:
            fd, tmp = tempfile.mkstemp(suffix=".img")
            with os.fdopen(fd, "wb") as f:
                f.write(media.data)
            path = tmp
        if not path or not os.path.isfile(path):
            stats.images_unavailable += 1
            return "[image: unavailable]"
        try:
            ext = extractor.extract_image(path, media.alt or None)
            desc = _one_line(ext.description or "")
        except Exception:  # noqa: BLE001 — one bad image must not fail the whole import
            desc = ""
        if desc:
            stats.images_perceived += 1
            return f"[image: {desc}]"
        # extractor ran but described nothing: fall back to the caption/placeholder
        return f"[image: {_one_line(media.alt) or 'image'}]"
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _one_line(text: str) -> str:
    return " ".join((text or "").split())
