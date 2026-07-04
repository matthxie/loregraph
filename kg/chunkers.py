"""Natural-boundary chunking — one source entry → several retrieval-grained episodes.

Strategy (chunk along the strongest structure the source already has, and only fall
back to semantics when there is none):

  * chat transcripts → conversation TURNS ("User: ..." / "Assistant: ..." line starts;
    the `[chat session — …]` header is re-prefixed onto every chunk so each keeps its
    temporal anchor).
  * everything else  → paragraphs (blank-line blocks).
  * either way, adjacent small units are PACKED up to `chunk_target_chars`, and a single
    oversized unit (a pasted document inside one chat turn, a wall-of-text paragraph) is
    split at paragraph → sentence boundaries under `chunk_max_chars`.

Chunk identity is DETERMINISTIC: a chunk's id is its ordinal under the parent
(`<source_id>#cNNN`), and its content hash keeps re-ingest idempotent — same input,
same chunks, same episodes, no duplicates. That is why splitting stays structural:
structural boundaries are reproducible across runs; embedding-drift boundaries are not.

The parent keeps the full original text on an un-rankable SOURCE node; chunks link to
it with PART_OF and to each other with NEXT (kg/ingest.py wires those edges).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TURN = re.compile(r"^(?:User|Assistant): ", re.MULTILINE)
_HEADER = re.compile(r"^\[chat session[^\]\n]*\]\s*\n")
_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    ordinal: int = 0


def _split_oversized(unit: str, max_chars: int) -> list[str]:
    """Break ONE oversized natural unit at paragraph, then sentence, boundaries.
    Deterministic and purely structural (no model in the loop)."""
    if len(unit) <= max_chars:
        return [unit]
    parts: list[str] = []
    for para in _PARA.split(unit):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            parts.append(para)
            continue
        # a single huge paragraph: sentence-pack under the ceiling
        buf = ""
        for sent in _SENT.split(para):
            if buf and len(buf) + len(sent) + 1 > max_chars:
                parts.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            parts.append(buf)
    return parts or [unit[:max_chars]]


def _pack(units: list[str], target: int, max_chars: int) -> list[str]:
    """Greedy-pack consecutive units up to ~target chars per chunk (keeps sequence,
    coalesces tiny units instead of emitting confetti)."""
    flat: list[str] = []
    for u in units:
        flat.extend(_split_oversized(u, max_chars))
    chunks: list[str] = []
    buf = ""
    for u in flat:
        if buf and len(buf) + len(u) + 1 > target:
            chunks.append(buf)
            buf = u
        else:
            buf = f"{buf}\n{u}".strip() if buf else u
    if buf:
        chunks.append(buf)
    return chunks


def _turn_units(text: str) -> tuple[str, list[str]]:
    """(session header, one string per conversation turn). Falls back to ("", [text])
    when the text carries no turn structure."""
    m = _HEADER.match(text)
    header = m.group(0).strip() if m else ""
    body = text[m.end():] if m else text
    starts = [mt.start() for mt in _TURN.finditer(body)]
    if not starts:
        return header, [body.strip()] if body.strip() else []
    units = []
    if starts[0] > 0 and body[:starts[0]].strip():
        units.append(body[:starts[0]].strip())
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(body)
        units.append(body[s:e].strip())
    return header, [u for u in units if u]


def chunk_text(text: str, *, target: int, max_chars: int) -> list[Chunk]:
    """Chunk one text entry along its natural boundaries. Returns [] when the text
    already fits in one chunk (caller ingests it unchunked, no parent node)."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return []
    if _TURN.search(text):
        header, units = _turn_units(text)
    else:
        header, units = "", [p.strip() for p in _PARA.split(text) if p.strip()]
    packed = _pack(units, target, max_chars)
    if len(packed) <= 1:
        return []          # no split happened — ingest unchunked, no parent node
    prefix = f"{header}\n" if header else ""
    return [Chunk(text=f"{prefix}{c}", ordinal=i) for i, c in enumerate(packed)]
