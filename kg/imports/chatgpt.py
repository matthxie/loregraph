"""ChatGPT export → canonical Conversations (the richest, hardest source).

A ChatGPT `conversations.json` stores each conversation as a `mapping` node tree, not a
flat list: user edits and assistant regenerations fork the tree, and only the path from
`current_node` back to the root is the conversation as it actually stands. We linearize
that active path (walk parents from current_node, reverse) which inherently DROPS the
abandoned edit/regeneration branches — they are not ancestors of current_node.

Content is routed by `content_type`:
  * text                → text
  * code                → text, is_code_hint=True (source-labeled code)
  * execution_output    → text (code-interpreter result)
  * multimodal_text     → text parts inline; each image_asset_pointer resolved to the
                          bundled media file → Media
Browsing chrome (tether_* content, browser tool calls) is skipped. `tool` role survives
ONLY for code-interpreter output, not for the browsing tool.

Unofficial/drifting format: an unknown block fails soft (skipped), never the whole file.
"""
from __future__ import annotations

import glob
import os
import re

from .canonical import Conversation, Media, Message
from .timeutil import to_iso

# content_type prefixes that are browsing/tool chrome, not conversation content.
_CHROME_PREFIXES = ("tether_browsing", "tether_quote", "system_error")


def to_conversations(data: object, base_dir: str) -> list[Conversation]:
    convs: list[Conversation] = []
    items = data if isinstance(data, list) else _unwrap(data)
    for raw in items:
        if not isinstance(raw, dict) or not isinstance(raw.get("mapping"), dict):
            continue                                    # not a conversation node — skip soft
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
    mapping = raw["mapping"]
    conv_id = str(raw.get("conversation_id") or raw.get("id") or _title_id(raw))
    conv = Conversation(id=f"chatgpt-{conv_id}",
                        title=str(raw.get("title") or ""),
                        created_at=to_iso(raw.get("create_time")))
    for node_id in _active_path(mapping, raw.get("current_node")):
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            continue
        msg = _message(node.get("message"), base_dir)
        if msg is not None:
            conv.messages.append(msg)
    if conv.created_at is None and conv.messages:
        conv.created_at = conv.messages[0].created_at
    return conv


def _title_id(raw: dict) -> str:
    return re.sub(r"\s+", "-", str(raw.get("title") or "untitled").strip().lower())[:60]


def _active_path(mapping: dict, current_node: str | None) -> list[str]:
    """Node ids from root → current_node along the ACTIVE branch. Walking parents up from
    current_node and reversing keeps only the surviving path, dropping every abandoned
    edit/regeneration sibling. Falls back to the whole mapping (root-first) if current_node
    is missing or the parent chain is broken."""
    if not current_node or current_node not in mapping:
        return _fallback_order(mapping)
    chain: list[str] = []
    seen: set[str] = set()
    node_id: str | None = current_node
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        chain.append(node_id)
        parent = mapping[node_id].get("parent")
        node_id = parent if isinstance(parent, str) else None
    chain.reverse()
    return chain


def _fallback_order(mapping: dict) -> list[str]:
    """Best-effort root-first traversal when current_node is unusable: find the root (no
    parent) and DFS the first child at each step. A drifting export that lost current_node
    still yields a readable linear thread instead of nothing."""
    roots = [nid for nid, n in mapping.items()
             if isinstance(n, dict) and not n.get("parent")]
    order: list[str] = []
    seen: set[str] = set()
    stack = list(reversed(roots)) or list(mapping.keys())
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in mapping:
            continue
        seen.add(nid)
        order.append(nid)
        kids = mapping[nid].get("children") or []
        stack.extend(reversed([k for k in kids if isinstance(k, str)]))
    return order


def _message(message: object, base_dir: str) -> Message | None:
    if not isinstance(message, dict):
        return None
    role = str((message.get("author") or {}).get("role") or "user").lower()
    content = message.get("content")
    if not isinstance(content, dict):
        return None
    ctype = str(content.get("content_type") or "text")
    if any(ctype.startswith(p) for p in _CHROME_PREFIXES):
        return None                                     # browsing/tool chrome
    meta = message.get("metadata") or {}
    if role == "system" and meta.get("is_user_system_message") is not True \
            and not content.get("parts") and ctype == "text":
        return None                                     # hidden/empty system scaffolding
    if role == "tool" and _is_browsing_tool(message):
        return None                                     # keep tool ONLY for code-interpreter

    created = to_iso(message.get("create_time"))
    if ctype == "code":
        text = _join_parts(content.get("parts"))
        return Message(role=_role(role), created_at=created, text=text, is_code_hint=True)
    if ctype in ("text", "execution_output"):
        text = _join_parts(content.get("parts") or [content.get("text", "")])
        if not text.strip():
            return None
        return Message(role=_role(role), created_at=created, text=text,
                       is_code_hint=(ctype == "execution_output"))
    if ctype == "multimodal_text":
        text, media = _multimodal(content.get("parts") or [], base_dir)
        if not text.strip() and not media:
            return None
        return Message(role=_role(role), created_at=created, text=text, media=media)
    # unknown content_type: keep any stringy parts, else skip (fail soft)
    text = _join_parts(content.get("parts"))
    return Message(role=_role(role), created_at=created, text=text) if text.strip() else None


def _is_browsing_tool(message: dict) -> bool:
    name = str((message.get("author") or {}).get("name") or "").lower()
    return "browser" in name or "web" in name


def _role(role: str) -> str:
    return role if role in ("user", "assistant", "system", "tool") else "user"


def _join_parts(parts: object) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""
    out = [p for p in parts if isinstance(p, str)]
    return "\n".join(out).strip()


def _multimodal(parts: list, base_dir: str) -> tuple[str, list[Media]]:
    """Split a multimodal_text parts[] into inline text and resolved image Media."""
    texts: list[str] = []
    media: list[Media] = []
    for part in parts:
        if isinstance(part, str):
            if part.strip():
                texts.append(part)
        elif isinstance(part, dict) and part.get("content_type") == "image_asset_pointer":
            resolved = _resolve_asset(part.get("asset_pointer") or "", base_dir)
            media.append(Media(kind="image", path=resolved,
                               alt=str(part.get("alt") or "")))
    return "\n".join(texts).strip(), media


def _resolve_asset(pointer: str, base_dir: str) -> str | None:
    """Resolve a `file-service://file-XYZ` (or `sediment://…file-XYZ`) pointer to the
    bundled export file. ChatGPT names the file `file-XYZ-<original name>`, so we glob the
    directory for `file-XYZ*`. Returns None when nothing matches (bytes not in the export →
    normalize emits an unavailable placeholder)."""
    if not pointer or not base_dir:
        return None
    # Strip the `file-service://` (or `sediment://…`) scheme first, else the regex would
    # greedily match the scheme's own "file-service" token instead of the real file id.
    tail = pointer.split("://")[-1]
    m = re.search(r"(file[-_][A-Za-z0-9]+)", tail)
    if not m:
        return None
    stem = m.group(1)
    for cand in sorted(glob.glob(os.path.join(base_dir, f"{stem}*"))):
        if os.path.isfile(cand):
            return cand
    return None
