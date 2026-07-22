"""Cold-start chat-history import (BUILD BRIEF).

Bulk-import a chat-history export from exactly ONE of three supported sources — ChatGPT,
Claude, or Gemini — into the unified graph. The public surface is a single facade method,
`Engine.import_conversations` (kg/engine.py); this package holds the per-source complexity
so the facade stays thin and delegates, exactly like ingest_repo/kg/code/.

Pipeline: detect (sniff) → per-source mapper (export → canonical Conversations) →
normalize (Conversations → session CorpusItems, images perceived inline) → the engine's
existing ingest path (source-agnostic from here on).
"""
from __future__ import annotations

from . import chatgpt, claude, gemini
from .canonical import Conversation, Media, Message
from .detect import UNRECOGNIZED, detect_from_data, load_export
from .normalize import NormalizeStats, to_corpus_items

# Closed, validated source set. Adding a 4th source is a new mapper here — not a change to
# the facade signature.
SUPPORTED_SOURCES = ("chatgpt", "claude", "gemini")

_MAPPERS = {
    "chatgpt": chatgpt.to_conversations,
    "claude": claude.to_conversations,
    "gemini": gemini.to_conversations,
}


def build_corpus_items(path: str, source: str, extractor):
    """Resolve an export at `path` for `source` ("auto" sniffs) into ingestible session
    CorpusItems. Returns (resolved_source, conversations, items, stats).

    `source` is assumed already validated against {auto} ∪ SUPPORTED_SOURCES by the caller
    (the facade); a value that reaches here outside that set is a programming error."""
    data, base_dir = load_export(path)
    resolved = detect_from_data(data) if source == "auto" else source
    conversations = _MAPPERS[resolved](data, base_dir)
    stats = NormalizeStats()
    items = to_corpus_items(conversations, extractor, stats)
    return resolved, conversations, items, stats


__all__ = [
    "SUPPORTED_SOURCES", "UNRECOGNIZED", "Conversation", "Message", "Media",
    "NormalizeStats", "build_corpus_items", "detect_from_data", "load_export",
    "to_corpus_items",
]
