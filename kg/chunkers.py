"""Natural-boundary chunking — one source entry → several retrieval-grained episodes.

Strategy (chunk along the strongest structure the source already has, and only fall
back to semantics when there is none):

  * chat transcripts → conversation TURNS ("User: ..." / "Assistant: ..." line starts;
    the `[chat session — …]` header is re-prefixed onto every chunk so each keeps its
    temporal anchor).
  * markdown → #/##/### heading sections, each chunk opened with its heading
    breadcrumb path (chunk_markdown).
  * code → top-level blank-line-delimited blocks (chunk_code, no AST).
  * everything else  → paragraphs (blank-line blocks; chunk_prose).
  * "auto" mode sniffs the format per entry (sniff_format) and routes accordingly
    via chunk_for.
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


def _pack(units: list[str], target: int, max_chars: int, sep: str = "\n") -> list[str]:
    """Greedy-pack consecutive units up to ~target chars per chunk (keeps sequence,
    coalesces tiny units instead of emitting confetti)."""
    flat: list[str] = []
    for u in units:
        flat.extend(_split_oversized(u, max_chars))
    chunks: list[str] = []
    buf = ""
    for u in flat:
        if buf and len(buf) + len(u) + len(sep) > target:
            chunks.append(buf)
            buf = u
        else:
            buf = f"{buf}{sep}{u}".strip() if buf else u
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


# --------------------------------------------------------------------------- #
# Phase 2 — format sniffer + multi-format chunkers. All deterministic (regex +
# arithmetic only — no LLM, no randomness): identical input → identical chunks.
# --------------------------------------------------------------------------- #
_MD_HEADING = re.compile(r"^#{1,6} \S", re.MULTILINE)
_MD_HEADING_LINE = re.compile(r"^(#{1,3}) \S")   # chunk boundaries: #, ##, ### only
_MD_FENCE = re.compile(r"^```", re.MULTILINE)
_MD_LIST = re.compile(r"^ {0,3}(?:[-*+]|\d+[.)]) ", re.MULTILINE)
_CODE_LINE = re.compile(
    r"^\s*(?:def |class |import |from \S+ import |function |const |let |var |"
    r"return\b|if __name__|#include\b|package |fn |func |public |private )",
    re.MULTILINE)


def sniff_format(text: str) -> str:
    """Classify one text entry as "turns" | "code" | "markdown" | "prose".

    Ordered heuristics — FIRST match wins, so the precedence is:
      1. turns    — >= 2 chat-turn line starts ("User: " / "Assistant: ").
      2. code     — shebang first line, OR >= 2 keyword lines (def/class/import/…)
                    AND keyword+codey lines (indented / brace / semicolon endings)
                    covering >= max(3, 30%) of non-blank lines. Checked BEFORE
                    markdown, so a source file whose comments look like headings
                    still lands here; the cost is that a markdown doc that is
                    mostly fenced code is classified as code (acceptable — code
                    chunking degrades gracefully on it).
      3. markdown — any # heading line, a ``` fence, or >= 3 list-marker lines.
      4. prose    — everything else (empty text included).
    """
    text = (text or "").strip()
    if not text:
        return "prose"
    if len(_TURN.findall(text)) >= 2:
        return "turns"
    if text.startswith("#!"):
        return "code"
    lines = [ln for ln in text.split("\n") if ln.strip()]
    kw = len(_CODE_LINE.findall(text))
    codey = sum(1 for ln in lines
                if ln.startswith(("    ", "\t")) or ln.rstrip().endswith(("{", "}", ";")))
    if kw >= 2 and (kw + codey) >= max(3, 0.3 * len(lines)):
        return "code"
    if _MD_HEADING.search(text) or _MD_FENCE.search(text) or len(_MD_LIST.findall(text)) >= 3:
        return "markdown"
    return "prose"


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """(breadcrumb, body) per heading-delimited section. The breadcrumb is the heading
    path down to and INCLUDING the section's own heading, joined with " > "
    (e.g. "# Guide > ## Setup"); the heading line itself moves into the breadcrumb, so
    bodies carry only content. Headings inside ``` fences are body text, not boundaries.
    Text before the first heading becomes a section with an empty breadcrumb."""
    sections: list[tuple[str, str]] = []
    levels: dict[int, str] = {}    # heading level → its heading line (current path)
    crumb = ""
    body: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            body.append(line)
            continue
        m = None if in_fence else _MD_HEADING_LINE.match(line)
        if not m:
            body.append(line)
            continue
        joined = "\n".join(body).strip()
        if joined or crumb:
            sections.append((crumb, joined))
        body = []
        lvl = len(m.group(1))
        levels = {k: v for k, v in levels.items() if k < lvl}
        levels[lvl] = line.rstrip()
        crumb = " > ".join(levels[k] for k in sorted(levels))
    joined = "\n".join(body).strip()
    if joined or crumb:
        sections.append((crumb, joined))
    return sections


def _pack_sections(sections: list[tuple[str, str]], target: int, max_chars: int) -> list[str]:
    """Pack (breadcrumb, body) sections like _pack, but keep every chunk self-describing:
    a chunk always OPENS with the breadcrumb of its first section (the markdown analogue
    of the turn chunker re-prefixing the session header — an oversized section split
    across chunks re-opens each one with its breadcrumb), and a section that starts
    mid-chunk carries its own breadcrumb line inline. Breadcrumb chars count against
    the budget, so max_chars stays a hard ceiling."""
    flat: list[tuple[str, str]] = []
    for crumb, body in sections:
        body_max = max(1, max_chars - len(crumb) - 1) if crumb else max_chars
        pieces = _split_oversized(body, body_max) if body else [""]
        flat.extend((crumb, p) for p in pieces)

    def render(crumb: str, piece: str) -> str:
        if not crumb:
            return piece
        return f"{crumb}\n{piece}" if piece else crumb

    chunks: list[str] = []
    buf = ""
    prev_crumb: str | None = None
    for crumb, piece in flat:
        part = piece if (crumb == prev_crumb and piece) else render(crumb, piece)
        if not part:
            prev_crumb = crumb
            continue
        if buf and len(buf) + len(part) + 1 > target:
            chunks.append(buf)
            buf = render(crumb, piece)     # new chunk re-opens with its breadcrumb
        else:
            buf = f"{buf}\n{part}" if buf else part
        prev_crumb = crumb
    if buf:
        chunks.append(buf)
    return chunks


def chunk_markdown(text: str, *, target: int, max_chars: int) -> list[Chunk]:
    """Split on #/##/### heading boundaries, pack adjacent sections to ~target chars
    under the max_chars ceiling (giant sections fall back paragraph→sentence). Every
    chunk opens with its heading breadcrumb path so it stays self-describing."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return []
    packed = _pack_sections(_markdown_sections(text), target, max_chars)
    if len(packed) <= 1:
        return []
    return [Chunk(text=c, ordinal=i) for i, c in enumerate(packed)]


def chunk_prose(text: str, *, target: int, max_chars: int) -> list[Chunk]:
    """Paragraph packing with sentence fallback for oversized paragraphs — the turn
    chunker's non-chat fallback, promoted to a first-class strategy for plain text."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return []
    units = [p.strip() for p in _PARA.split(text) if p.strip()]
    packed = _pack(units, target, max_chars)
    if len(packed) <= 1:
        return []
    return [Chunk(text=c, ordinal=i) for i, c in enumerate(packed)]


def _code_units(text: str) -> list[str]:
    """Top-level blank-line-delimited blocks: a blank-line run closes a block only when
    the next non-blank line starts at column 0, so a function/class body's internal
    blank lines never split it and top-level definitions fall out naturally."""
    lines = text.split("\n")
    blocks: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip():
            buf.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            break
        if buf and not lines[j][0].isspace():
            blocks.append("\n".join(buf))
            buf = []
        elif buf:
            buf.extend(lines[i:j])     # blank line(s) inside an indented body
        i = j
    if buf:
        blocks.append("\n".join(buf))
    return [b for b in blocks if b.strip()]


def chunk_code(text: str, *, target: int, max_chars: int) -> list[Chunk]:
    """Minimal v1 code chunker: pack top-level blank-line-delimited blocks to budget
    (no AST). Blocks are re-joined with a blank line to keep definitions readable."""
    text = (text or "").strip("\n")
    if len(text.strip()) <= max_chars:
        return []
    packed = _pack(_code_units(text), target, max_chars, sep="\n\n")
    if len(packed) <= 1:
        return []
    return [Chunk(text=c, ordinal=i) for i, c in enumerate(packed)]


_CHUNKERS = {"turns": chunk_text, "markdown": chunk_markdown,
             "prose": chunk_prose, "code": chunk_code}


def chunk_for(text: str, *, mode: str, target: int, max_chars: int) -> list[Chunk]:
    """Dispatch one text entry to the right chunker. mode "auto" sniffs the format per
    entry; "turns" is exactly the original chunk_text behavior; unknown modes fall back
    to chunk_text. Every chunker honors the same contract: [] when no split is needed,
    deterministic ordinals otherwise."""
    if mode == "auto":
        mode = sniff_format(text)
    return _CHUNKERS.get(mode, chunk_text)(text, target=target, max_chars=max_chars)
