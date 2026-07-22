"""Canonical chat-history schema — the one shape every per-source mapper emits so the
normalize path (kg/imports/normalize.py) is source-agnostic.

A ChatGPT / Claude / Gemini export is mapped into `Conversation` → `Message` → `Media`;
after that single funnel nothing downstream knows or cares which app produced it. The
schema is deliberately lossy-but-faithful: only what the graph can actually use (role,
time, text, resolved image attachments, a code hint) survives; app-specific chrome
(browsing tethers, tool plumbing, model metadata) is dropped by the mapper, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Roles the graph understands. A mapper must coerce its source's sender label into one of
# these; anything unknown collapses to "user" (never dropped — a turn with no role is still
# a turn) except explicit tool/system output, which keeps its label so normalize can format
# it distinctly.
ROLES = ("user", "assistant", "system", "tool")


@dataclass
class Media:
    """One resolved attachment on a message. `path` points at a bundled export file (the
    mapper has already resolved the source's opaque pointer to it); `data` carries raw bytes
    when the export inlines them instead. `alt` is any source-supplied caption/description.
    Exactly one of path/data is expected to be truthy for a perceivable image; when neither
    is, normalize emits an `[image: unavailable]` placeholder rather than perceiving."""
    kind: str = "image"                 # only "image" is perceived today
    path: str | None = None             # resolved bundled-file path
    data: bytes | None = None           # inlined bytes (written to a temp file to perceive)
    alt: str = ""                       # source caption / filename, best-effort


@dataclass
class Message:
    role: str = "user"                  # one of ROLES
    created_at: str | None = None       # ISO-8601 (mapper-normalized); None → inherit session
    text: str = ""                      # the message's plain text (code kept inline as text)
    media: list[Media] = field(default_factory=list)
    is_code_hint: bool = False          # source labeled this content as code (ChatGPT only)


@dataclass
class Conversation:
    id: str                             # the source's conversation id (stable across re-runs)
    title: str = ""
    created_at: str | None = None       # ISO-8601; None → derived from the first message
    messages: list[Message] = field(default_factory=list)
