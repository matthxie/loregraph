"""Obsidian vault → ingestible notes + a wikilink/tag link-spec (BUILD BRIEF).

A vault is already a personal knowledge graph — the human has curated the links (wikilinks),
the topics (tags), and the identity (frontmatter aliases). So unlike the chat importers, which
must INFER structure from prose, this importer LEANS on the explicit structure the author
authored: one `.md` file → one text Episode, `[[wikilinks]]` → deterministic Episode→Episode
HYPERLINKS_TO edges, `#tags` / frontmatter `tags:` → deterministic TAGGED_AS, and `aliases:` →
extra resolver keys so `[[an alias]]` still finds its note.

This module owns the vault-shaped complexity (walk, frontmatter/wikilink/tag/embed parsing,
Obsidian's link-resolution rules) and hands the engine two flat things:
  * `to_corpus_items(notes, extractor, stats, perceive=…)` — notes → text CorpusItems, with
    `![[image.png]]` / `![](img)` embeds perceived-and-inlined as `[image: …]` (reusing the
    conversation importer's `_perceive` helper, since vault images are always local bytes);
  * `build_resolver(notes)` — the name/path/alias → note map that pass 2 uses to turn each
    note's wikilinks into real Episode ids.

It is NOT a conversation source: it does not go through canonical.py / normalize.py (those are
message-shaped). Two-pass wikilink resolution (you can't point `[[B]]` at B's Episode until B is
ingested) lives in the engine facade (Engine.import_vault); everything parsing-shaped is here.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

import yaml

from ..corpus import CorpusItem
from .canonical import Media
from .normalize import NormalizeStats, _one_line, _perceive
from .timeutil import to_iso

# Attachment extensions treated as perceivable images (an `![[x.png]]` embed is inlined via
# the VLM); anything else embedded via `![[Other Note]]` is a note transclusion → a wikilink.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".tif", ".tiff",
               ".heic", ".heif"}

# `[[target]]`, `[[target#heading]]`, `[[target|alias]]`, `![[embed]]` — one regex, the
# leading `!` (group 1) marking an embed/transclusion vs a plain link.
_WIKILINK = re.compile(r"(!?)\[\[([^\[\]]+?)\]\]")
# Standard-markdown image: `![alt](path)`. A remote `http(s)://` target is out of scope
# (non-local, deferred with PDFs) — only vault-local relative paths are perceived.
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
# An inline `#tag`: `#` + a letter, then tag chars (Obsidian allows `-`, `_`, `/` nesting).
# The lookbehind rejects `word#frag` (URL fragments) and `##heading`; the leading-letter
# rule rejects `#123` (pure-numeric is not a tag) and `# Heading` (space after `#`).
_INLINE_TAG = re.compile(r"(?<![\w#/])#([A-Za-z][\w/-]*)")
# A fenced code block (``` … ```): stripped before tag/link scanning so a `#comment` or a
# `[[x]]` sitting in a code sample is not mistaken for real structure.
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class WikiLink:
    """One `[[target]]` / `![[target]]` occurrence in a note body. `target` is the bare note
    name with any `#heading` / `|alias` display suffix already stripped — what the resolver
    keys on. `is_embed` distinguishes `![[Note]]` transclusion (still just a HYPERLINKS_TO
    edge for us) from a plain `[[Note]]` link; both wire identically in pass 2."""
    target: str
    is_embed: bool = False


@dataclass
class Note:
    """One parsed `.md` file. `body` is the frontmatter-stripped text (the Episode's raw_text);
    `item_id` is the deterministic CorpusItem id derived from the vault-relative path, so the
    Episode id (`ep_<item_id>`) is stable across re-imports and lets pass 2 find it."""
    abspath: str
    rel_path: str                        # vault-relative → the Episode's source_ref
    item_id: str                         # deterministic corpus id (stable per rel_path)
    title: str
    body: str
    created_at: str | None
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    wikilinks: list[WikiLink] = field(default_factory=list)
    image_embeds: list[tuple[str, str]] = field(default_factory=list)  # (raw token, abspath|"")


def _item_id(rel_path: str) -> str:
    """Deterministic, filesystem-agnostic CorpusItem id for a vault path. A hash keeps it
    stable and collision-free regardless of spaces/unicode/depth in the note path, so a
    re-import lands on the SAME id (→ same Episode, hash-cache dedup) and wikilink targets
    resolve identically run to run."""
    return "obs_" + hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]


def is_vault(path: str) -> bool:
    """A vault is a directory of markdown — usually holding an `.obsidian/` config dir, but we
    don't require it (a plain folder of `.md` notes is a perfectly good import target). Reject
    a non-directory or a directory with no `.md` anywhere so a wrong path fails loudly."""
    if not os.path.isdir(path):
        return False
    if os.path.isdir(os.path.join(path, ".obsidian")):
        return True
    for _root, _dirs, files in _walk(path):
        if any(f.lower().endswith(".md") for f in files):
            return True
    return False


def _walk(path: str):
    """os.walk that prunes Obsidian's own dot-dirs (`.obsidian`, `.trash`) and any hidden
    directory — plugin config / version-control noise is never note content."""
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        yield root, dirs, files


def _asset_index(path: str) -> dict[str, str]:
    """basename(lower) → abspath for every NON-markdown file in the vault, so an `![[img.png]]`
    embed (which names only the basename, Obsidian resolving it vault-wide) finds its bytes.
    First writer wins on a basename collision — good enough; exact-path embeds bypass this."""
    idx: dict[str, str] = {}
    for root, _dirs, files in _walk(path):
        for f in files:
            if f.lower().endswith(".md"):
                continue
            idx.setdefault(f.lower(), os.path.join(root, f))
    return idx


def parse_vault(path: str) -> list[Note]:
    """Walk `*.md` under the vault → parsed `Note`s (sorted by rel_path for deterministic
    ingest order). One malformed note (bad YAML, unreadable bytes) is skipped-soft, never
    sinking the import."""
    assets = _asset_index(path)
    notes: list[Note] = []
    for root, _dirs, files in _walk(path):
        for f in sorted(files):
            if not f.lower().endswith(".md"):
                continue
            abspath = os.path.join(root, f)
            rel = os.path.relpath(abspath, path).replace(os.sep, "/")
            try:
                note = _parse_note(abspath, rel, assets)
            except Exception:  # noqa: BLE001 — one bad note must not sink the vault import
                continue
            if note is not None:
                notes.append(note)
    notes.sort(key=lambda n: n.rel_path)
    return notes


def _parse_note(abspath: str, rel: str, assets: dict[str, str]) -> Note | None:
    with open(abspath, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    meta, body = _split_frontmatter(raw)

    # created_at: frontmatter date/created (author's own dating) wins; else the file's ctime
    # so every note is still ordered on the bi-temporal axis.
    created = _fm_date(meta) or to_iso(os.stat(abspath).st_ctime)

    title = _title(body, abspath)
    aliases = _fm_list(meta.get("aliases"))
    # tags come from BOTH frontmatter `tags:` and inline `#tags`, de-duped, `#` stripped.
    scan = _strip_code(body)
    tags = _dedup(_fm_list(meta.get("tags")) + _inline_tags(scan))
    wikilinks = _wikilinks(scan)
    image_embeds = _image_embeds(scan, abspath, assets)

    return Note(abspath=abspath, rel_path=rel, item_id=_item_id(rel), title=title,
                body=body, created_at=created, tags=tags, aliases=aliases,
                wikilinks=wikilinks, image_embeds=image_embeds)


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    """Peel a leading `--- … ---` YAML block. Parse fail-soft: malformed YAML (or a non-dict
    document) yields an empty meta + the body AFTER the block, so a broken header degrades to
    a plain note rather than crashing the walk."""
    if not raw.startswith("---"):
        return {}, raw
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n?(.*)$", raw, re.DOTALL)
    if not m:
        return {}, raw
    block, body = m.group(1), m.group(2)
    try:
        meta = yaml.safe_load(block)
    except Exception:  # noqa: BLE001 — malformed frontmatter must not sink the note
        meta = None
    return (meta if isinstance(meta, dict) else {}), body


def _fm_date(meta: dict) -> str | None:
    for key in ("date", "created", "created_at", "date created"):
        if meta.get(key) not in (None, ""):
            iso = to_iso(str(meta[key]))
            if iso:
                return iso
    return None


def _fm_list(value: object) -> list[str]:
    """Frontmatter list-ish field (`tags:`/`aliases:`) → clean surface list. YAML may hand us
    a list, or a single scalar, or a comma/space string (`tags: a, b c`) — all normalized;
    a leading `#` on a tag is stripped so `#ml` and `ml` share one surface."""
    out: list[str] = []
    if value is None:
        return out
    items = value if isinstance(value, (list, tuple)) else re.split(r"[,\s]+", str(value))
    for it in items:
        s = str(it).strip().lstrip("#").strip()
        if s:
            out.append(s)
    return out


def _title(body: str, abspath: str) -> str:
    m = _H1.search(body)
    if m:
        return m.group(1).strip()
    return os.path.splitext(os.path.basename(abspath))[0]


def _strip_code(text: str) -> str:
    """Blank out fenced + inline code so a `#tag`-looking comment or a `[[x]]`-looking token
    inside a code sample is not scanned as real structure. Positions don't matter (we only
    read tokens out), so we replace with spaces of no particular length."""
    text = _FENCE.sub(" ", text)
    return _INLINE_CODE.sub(" ", text)


def _inline_tags(text: str) -> list[str]:
    return [m.group(1) for m in _INLINE_TAG.finditer(text)]


def _wikilink_target(inner: str) -> str:
    """`folder/Note#heading|Display` → `folder/Note`: drop the `|alias` display text and the
    `#heading` anchor, keeping just the note name the resolver keys on."""
    return inner.split("|", 1)[0].split("#", 1)[0].strip()


def _wikilinks(text: str) -> list[WikiLink]:
    """Every `[[…]]` / `![[…]]` that is NOT an image embed. An `![[x.png]]` embed is an image
    (handled by _image_embeds); an `![[Other Note]]` embed is a transclusion → still a link."""
    out: list[WikiLink] = []
    for m in _WIKILINK.finditer(text):
        is_embed = m.group(1) == "!"
        target = _wikilink_target(m.group(2))
        if not target:
            continue
        if is_embed and _has_image_ext(target):
            continue                                  # an image embed, not a note link
        out.append(WikiLink(target=target, is_embed=is_embed))
    return out


def _has_image_ext(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in _IMAGE_EXTS


def _image_embeds(text: str, note_abspath: str, assets: dict[str, str]) \
        -> list[tuple[str, str]]:
    """All image embeds in a note → (raw token, resolved abspath). Two syntaxes:
    `![[image.png]]` (Obsidian, basename resolved vault-wide via the asset index) and
    `![alt](path)` (markdown, path resolved relative to the note; remote URLs skipped).
    An unresolved embed keeps abspath="" → the caller inlines an `[image: <name>]` placeholder
    (never a crash), same fail-soft spirit as the conversation importer's missing-bytes path."""
    out: list[tuple[str, str]] = []
    note_dir = os.path.dirname(note_abspath)
    for m in _WIKILINK.finditer(text):
        if m.group(1) != "!":
            continue
        target = _wikilink_target(m.group(2))
        if not _has_image_ext(target):
            continue
        abspath = assets.get(os.path.basename(target).lower(), "")
        out.append((m.group(0), abspath))
    for m in _MD_IMAGE.finditer(text):
        ref = m.group(2).strip().split(" ")[0]        # drop any `"title"` suffix
        if ref.startswith(("http://", "https://", "data:")):
            continue                                  # remote / inline-data: out of scope
        cand = os.path.normpath(os.path.join(note_dir, ref))
        abspath = cand if os.path.isfile(cand) else assets.get(os.path.basename(ref).lower(), "")
        out.append((m.group(0), abspath))
    return out


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        k = it.lower()
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


# --------------------------------------------------------------------------- #
# Resolver (Obsidian's link-resolution rules)
# --------------------------------------------------------------------------- #
class Resolver:
    """Maps a wikilink target → the note's `item_id`, applying Obsidian's rules: `[[Foo]]`
    matches `Foo.md` anywhere in the vault, case-insensitively; a `folder/Foo` target matches
    by relative path; ambiguous bare names disambiguate to the SHALLOWEST match (fewest path
    segments); and a note is also reachable by any of its frontmatter `aliases`."""

    def __init__(self, notes: list[Note]):
        self._by_path: dict[str, str] = {}     # rel-path-noext(lower) → item_id
        self._by_name: dict[str, str] = {}     # basename-noext(lower) → item_id (shallowest)
        self._by_alias: dict[str, str] = {}    # alias(lower) → item_id
        depth: dict[str, int] = {}
        for n in notes:
            rel_noext = n.rel_path[:-3].lower() if n.rel_path.lower().endswith(".md") \
                else n.rel_path.lower()
            self._by_path.setdefault(rel_noext, n.item_id)
            base = rel_noext.rsplit("/", 1)[-1]
            d = n.rel_path.count("/")
            if base not in self._by_name or d < depth.get(base, 1 << 30):
                self._by_name[base] = n.item_id    # shallower path wins ambiguous bare names
                depth[base] = d
            for a in n.aliases:
                self._by_alias.setdefault(a.strip().lower(), n.item_id)

    def resolve(self, target: str) -> str | None:
        low = target.strip().lower()
        if low.endswith(".md"):
            low = low[:-3]
        if not low:
            return None
        if "/" in low:                              # path-qualified: `folder/Note`
            return self._by_path.get(low) or self._by_name.get(low.rsplit("/", 1)[-1])
        return self._by_name.get(low) or self._by_alias.get(low) or self._by_path.get(low)


def build_resolver(notes: list[Note]) -> Resolver:
    return Resolver(notes)


# --------------------------------------------------------------------------- #
# Notes → CorpusItems (with perceive-and-inline images)
# --------------------------------------------------------------------------- #
def to_corpus_items(notes: list[Note], extractor, stats: NormalizeStats | None = None,
                    perceive: bool = True) -> list[CorpusItem]:
    """Each `Note` → one text CorpusItem whose text is the note body with every image embed
    replaced in place by an `[image: <description>]` marker, so the image content is part of
    the Episode's raw_text + embedding surface (one Episode per note, no extra plumbing).

    `perceive=True` runs the VLM on each image (reusing the conversation importer's `_perceive`
    → the SAME `[image: …]` shape); `perceive=False` (structure-only mode) skips the LLM and
    inlines an `[image: <filename/alt>]` placeholder instead, so extract=False stays model-free.
    """
    stats = stats or NormalizeStats()
    items: list[CorpusItem] = []
    for n in notes:
        text = _render_body(n, extractor, stats, perceive)
        items.append(CorpusItem(
            id=n.item_id, modality="text", source_ref=n.rel_path,
            title=n.title, text=text, created_at=n.created_at))
    return items


def _render_body(note: Note, extractor, stats: NormalizeStats, perceive: bool) -> str:
    body = note.body
    for token, abspath in note.image_embeds:
        marker = _image_marker(token, abspath, extractor, stats, perceive)
        body = body.replace(token, marker)
    return body.strip()


def _image_marker(token: str, abspath: str, extractor, stats: NormalizeStats,
                  perceive: bool) -> str:
    """One image embed → its inline `[image: …]` marker. With perception on and bytes present,
    the VLM describes it (via the shared `_perceive`); otherwise fall back to the embed's
    filename/alt so the note still records that an image was there."""
    name = _one_line(_embed_name(token)) or "image"
    if perceive and abspath:
        return _perceive(Media(kind="image", path=abspath, alt=name), extractor, stats)
    return f"[image: {name}]"


def _embed_name(token: str) -> str:
    """The human-readable name inside an embed token, for the placeholder path: `![[a/b.png]]`
    → `b.png`; `![alt](path)` → `alt` (or the path's basename)."""
    m = _WIKILINK.match(token)
    if m:
        return os.path.basename(_wikilink_target(m.group(2)))
    m = _MD_IMAGE.match(token)
    if m:
        return m.group(1).strip() or os.path.basename(m.group(2).strip().split(" ")[0])
    return ""


__all__ = ["Note", "WikiLink", "Resolver", "is_vault", "parse_vault", "build_resolver",
           "to_corpus_items"]
