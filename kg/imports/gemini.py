"""Gemini export → canonical Conversations (coarse Takeout fallback, lowest fidelity).

Gemini has no first-class chat export; the data lives in Google Takeout's "My Activity"
as flat activity records (one prompt + its response, tagged to the Gemini Apps product).
There is no conversation grouping, no reliable media, and no code typing — so this is a
best-effort reconstruction: each activity record becomes a tiny two-message Conversation
(a user prompt + an assistant response), text-only. Acceptable per the BUILD BRIEF —
Gemini is the coarse fallback, not a fidelity target.
"""
from __future__ import annotations

import re

from .canonical import Conversation, Message
from .timeutil import to_iso

# A "My Activity" prompt title is usually "Prompted <the text>" (or the raw text). Strip the
# leading verb so the user turn is the actual prompt.
_PROMPT_PREFIX = re.compile(r"^(?:Prompted|Asked|Searched for|Said)\s+", re.IGNORECASE)


def to_conversations(data: object, base_dir: str) -> list[Conversation]:
    records = data if isinstance(data, list) else _unwrap(data)
    convs: list[Conversation] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        conv = _one_record(rec, i)
        if conv is not None and conv.messages:
            convs.append(conv)
    return convs


def _unwrap(data: object) -> list:
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def _one_record(rec: dict, index: int) -> Conversation | None:
    header = str(rec.get("header", "")).lower()
    products = " ".join(str(x) for x in (rec.get("products") or [])).lower()
    if not any(tag in f"{header} {products}" for tag in ("gemini", "bard")):
        return None                                     # a non-Gemini Takeout row — skip soft
    created = to_iso(rec.get("time"))
    prompt = _prompt(rec)
    response = _response(rec)
    messages: list[Message] = []
    if prompt:
        messages.append(Message(role="user", created_at=created, text=prompt))
    if response:
        messages.append(Message(role="assistant", created_at=created, text=response))
    if not messages:
        return None
    # Takeout has no conversation id; the record's own index gives a stable, re-run-safe id.
    return Conversation(id=f"gemini-{index:06d}", title=prompt[:60],
                       created_at=created, messages=messages)


def _prompt(rec: dict) -> str:
    title = str(rec.get("title") or "").strip()
    return _PROMPT_PREFIX.sub("", title).strip()


def _response(rec: dict) -> str:
    """The model's reply, when Takeout captured it. Newer exports carry it in `subtitles`
    or a `details`/description field; older ones only have the prompt (response stays empty,
    a user-only turn — still a valid, if thin, episode)."""
    for key in ("subtitles",):
        blocks = rec.get(key)
        if isinstance(blocks, list):
            parts = [str(b.get("name")) for b in blocks
                     if isinstance(b, dict) and b.get("name")]
            if parts:
                return "\n".join(parts).strip()
    for key in ("description", "snippet"):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""
