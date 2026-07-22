"""Claude export → canonical Conversations (easy structure, untyped code).

A Claude `conversations.json` is far simpler than ChatGPT's: each conversation carries a
flat `chat_messages[]` (no branching tree — nothing to linearize or prune). `sender` maps
directly to role; text comes from the `content[]` text blocks (or the legacy top-level
`text`). Attachments/files land as image Media.

Caveats the BUILD BRIEF nails + real-export realities:
  * Code is plain text here — Claude does not label it — so is_code_hint stays False and the
    downstream heuristic sniffer decides code-ness.
  * `attachments[]` are usually PASTED TEXT DOCUMENTS, not images: Claude ships their
    `extracted_content` inline (a `file_type` of txt/md/pdf/csv/…, often with an empty
    file_name). Those are INLINED as text — dropping them (or worse, mislabeling them as an
    image) would lose the richest content a chat carries. Only genuine image attachments
    become Media.
  * `files[]` are opaque uuid references whose bytes are NOT in the export. An image-named one
    still becomes a Media (→ an `[image: unavailable]` placeholder — we never fabricate a
    perception); a non-image named one (e.g. a résumé PDF) is noted as `[attachment: name]`;
    an empty, contentless reference is skipped rather than mislabeled.
"""
from __future__ import annotations

import os

from .canonical import Conversation, Media, Message
from .timeutil import to_iso

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".heic", ".heif",
               ".tif", ".tiff"}
_IMAGE_TYPES = {"image", "jpg", "jpeg", "png", "webp", "gif", "bmp", "heic", "heif",
                "tif", "tiff"}


def _is_image(file_type: str, name: str) -> bool:
    if (file_type or "").lower() in _IMAGE_TYPES:
        return True
    return os.path.splitext(name or "")[1].lower() in _IMAGE_EXTS


def to_conversations(data: object, base_dir: str) -> list[Conversation]:
    convs: list[Conversation] = []
    items = data if isinstance(data, list) else _unwrap(data)
    for raw in items:
        if not isinstance(raw, dict) or not isinstance(raw.get("chat_messages"), list):
            continue
        conv = _one_conversation(raw, base_dir)
        if conv.messages:
            convs.append(conv)
    return convs


def _unwrap(data: object) -> list:
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def _one_conversation(raw: dict, base_dir: str) -> Conversation:
    conv_id = str(raw.get("uuid") or raw.get("id") or "untitled")
    conv = Conversation(id=f"claude-{conv_id}",
                        title=str(raw.get("name") or raw.get("title") or ""),
                        created_at=to_iso(raw.get("created_at")))
    for m in raw["chat_messages"]:
        if not isinstance(m, dict):
            continue
        msg = _message(m, base_dir)
        if msg is not None:
            conv.messages.append(msg)
    if conv.created_at is None and conv.messages:
        conv.created_at = conv.messages[0].created_at
    return conv


def _message(m: dict, base_dir: str) -> Message | None:
    sender = str(m.get("sender") or m.get("role") or "user").lower()
    role = "assistant" if sender in ("assistant", "model", "claude") else \
        "user" if sender in ("user", "human") else \
        sender if sender in ("system", "tool") else "user"
    text = _text(m)
    extra, media = _attachments(m, base_dir)
    full = "\n\n".join(t for t in [text, *extra] if t.strip())
    if not full.strip() and not media:
        return None
    return Message(role=role, created_at=to_iso(m.get("created_at")),
                   text=full, media=media)


def _text(m: dict) -> str:
    content = m.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t)
            elif isinstance(block, str) and block.strip():
                parts.append(block)
        if parts:
            return "\n".join(parts).strip()
    if isinstance(m.get("text"), str):
        return m["text"].strip()
    return ""


def _attachments(m: dict, base_dir: str) -> tuple[list[str], list[Media]]:
    """Split a message's attachments/files into inline text snippets and image Media.

    Returns (texts, media). A pasted text document (extracted_content present) is inlined
    verbatim; a genuine image becomes a Media (bytes resolved or an unavailable placeholder);
    any other named file is noted as `[attachment: name]`; an empty, contentless reference is
    dropped (it carries nothing usable)."""
    texts: list[str] = []
    media: list[Media] = []
    for ref in list(m.get("attachments") or []):
        if not isinstance(ref, dict):
            continue
        name = str(ref.get("file_name") or ref.get("name") or "").strip()
        ftype = str(ref.get("file_type") or "").strip()
        content = ref.get("extracted_content")
        if isinstance(content, str) and content.strip():
            head = f"[attachment: {name}]\n" if name else "[attachment]\n"
            texts.append(head + content.strip())
        elif _is_image(ftype, name):
            media.append(Media(kind="image", path=_resolve(ref, name, base_dir), alt=name))
        elif name:
            texts.append(f"[attachment: {name}]")
    for ref in list(m.get("files") or []):
        if not isinstance(ref, dict):
            continue
        name = str(ref.get("file_name") or ref.get("name") or "").strip()
        if _is_image("", name):
            media.append(Media(kind="image", path=_resolve(ref, name, base_dir), alt=name))
        elif name:
            texts.append(f"[attachment: {name}]")
        # else: an opaque uuid reference with no name/bytes — nothing to ingest, skip
    return texts, media


def _resolve(ref: dict, name: str, base_dir: str) -> str | None:
    for key in ("file_path", "path"):
        p = ref.get(key)
        if isinstance(p, str) and base_dir:
            cand = p if os.path.isabs(p) else os.path.join(base_dir, p)
            if os.path.isfile(cand):
                return cand
    if name and base_dir:
        for sub in ("", "attachments", "files"):
            cand = os.path.join(base_dir, sub, name)
            if os.path.isfile(cand):
                return cand
    return None
