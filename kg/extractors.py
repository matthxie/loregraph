"""Extractors (docs/ARCHITECTURE.md §6.4).

Pull `{entities[], tags[], relations[]}` directly from raw content (rev 2 — no
summary step) in one structured-output call, with an optional reflexion recall pass.
Relations are directed and carry open-vocabulary `labels[]` (rev 3) that are
consolidated into canonical relationship-tag nodes downstream.

  * OpenAIExtractor  — the real, live path: the active provider (make_client) forced
                       into a typed tool call, vision for images. The model is the
                       per-provider default unless config.llm_model was overridden.
  * ScriptedExtractor — a deterministic {text: Extraction} table used ONLY by the
                       synthetic temporal demo + unit tests (it stubs the LLM so the
                       graph's open/close/supersede logic runs on known facts). Not a
                       general extractor — unknown text yields an empty Extraction.

Both return the same `Extraction` object so the ingestion pipeline is backend-blind.
The old offline HeuristicExtractor (proper-noun + keyword guessing) was removed: it
produced low-quality entities/tags and a single `related_to` predicate, which is not
representative of the live graph. Extraction is now live-only.
"""
from __future__ import annotations

import base64
import html
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Protocol

from .backoff import call_with_backoff
from .config import Config
from .cues import cue_kinds, has_cue
from .errors import UnsupportedMedia
from .llm_client import llm_available, make_client, resolve_model
from .metering import UsageMeter
from .models import EntityCategory, EntityType, Provenance, entity_category_for_type
from .profiler import span as prof_span

# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class ExtractedEntity:
    name: str
    type: EntityType = EntityType.OTHER
    # Broad glyph category (person/place/thing) for graph clients. None = not stated;
    # the parse path fills it from the payload or entity_category_for_type(type), and
    # ingest passes it through entity_node(category=...).
    category: EntityCategory | None = None


@dataclass
class ExtractedRelation:
    """A directed connection source→target carrying open-vocabulary relationship
    labels (e.g. ["is_friend_of", "works_with"]). The labels are consolidated into
    canonical relationship-tag nodes downstream (Canonicalizer.resolve_relation).

    Temporal fields (docs/TEMPORAL.md §6): `status` is the polarity — "asserted" (the
    relationship holds), "ended" (it terminated / no longer holds, from cues like "former",
    "ex-", "no longer" → a valid-time CLOSE) or "retracted" (a correction: the relationship
    was NEVER true, the prior claim was a mistake → a belief flip, not a valid-time end).
    `valid_from` / `valid_to` are OPTIONAL stated bounds (ISO strings); empty means
    "unknown / as of this episode"."""
    source: str
    target: str
    labels: list[str] = field(default_factory=list)
    provenance: Provenance = Provenance.EXTRACTED
    confidence: float = 0.8
    status: str = "asserted"          # "asserted" | "ended" | "retracted"
    valid_from: str = ""
    valid_to: str = ""


# Process/meta tags that describe the ACT of note-taking / ingest plumbing rather than
# what a note is ABOUT. They leak in as generic themes ("shipped", "refactor", "routing"),
# bridge unrelated notes, and bury the personal signal, so they are dropped at parse time.
# Kept as a small module-level frozenset so it is easy to extend; matched case-insensitively
# against the trimmed tag surface.
TAG_STOPLIST = frozenset({
    "shipped", "resolution", "test", "input", "refactor", "documentation", "routing",
})


# Precedence when two duplicate relations disagree on polarity during a merge: a termination
# (ended / retracted — a stated valid-time close or belief flip) carries more information than
# a plain assertion, so it must not be silently overwritten by 'asserted'. ended and retracted
# are peers (neither dominates the other); either beats asserted.
_STATUS_RANK = {"asserted": 0, "ended": 1, "retracted": 1}


def _status_rank(status: str) -> int:
    return _STATUS_RANK.get((status or "asserted").lower(), 0)


@dataclass
class ExtractedFact:
    """A stated amount/count/measurement — the typed home for quantities (docs:
    extraction-completeness fix). Kept OUT of entities[]/relations[] so it is exempt from
    the entity salience filter ("do not pad") that was silently dropping every dollar
    amount before this existed. One instance == one OCCURRENCE: two mentions of the same
    subject/predicate/value/date are two facts, not one deduplicated relation."""
    subject: str
    predicate: str
    value: float
    unit: str = ""
    date: str = ""

    def dedup_key(self) -> tuple:
        return (self.subject.strip().lower(), self.predicate.strip().lower(),
                round(self.value, 6), self.unit.strip().lower(), self.date.strip())


@dataclass
class Extraction:
    entities: list[ExtractedEntity] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    description: str | None = None  # described media: the one-line VLM / page description
    # LINK ingest only: the full fetched page body, preserved verbatim so ingest can write an
    # un-rankable SOURCE provenance node (auditable + re-extractable) without embedding it.
    source_text: str | None = None
    # LINK ingest only: the fetched page <title>, so derive_title can prefer it (the real page
    # title) over a title derived from the subject description — carried here to avoid a 2nd fetch.
    page_title: str | None = None

    def merge(self, other: "Extraction") -> "Extraction":
        names = {e.name.lower() for e in self.entities}
        for e in other.entities:
            if e.name.lower() not in names:
                self.entities.append(e); names.add(e.name.lower())
        tagset = {t.lower() for t in self.tags}
        for t in other.tags:
            if t.lower() not in tagset:
                self.tags.append(t); tagset.add(t.lower())
        # merge by directed (source, target, valid_from): union label sets of TRUE
        # duplicates, but keep distinct-dated occurrences of the same pair/predicate
        # separate (docs: per-occurrence events must not collapse across a reflexion/
        # sectioning merge just because they share a source/target). A collision must
        # NOT silently discard the duplicate's temporal signal: a TERMINATION
        # (ended/retracted) wins over a plain 'asserted' — a section saying "no longer
        # works with X" must close the fact even if an earlier section asserted it —
        # and an empty valid_to is filled from the duplicate that states one
        # (valid_from is part of the merge key, so it is already equal within a key).
        by_pair: dict[tuple[str, str, str], ExtractedRelation] = {}
        for r in self.relations:
            by_pair[(r.source.lower(), r.target.lower(), r.valid_from)] = r
        for r in other.relations:
            kk = (r.source.lower(), r.target.lower(), r.valid_from)
            if kk in by_pair:
                have = by_pair[kk]
                for lab in r.labels:
                    if lab not in have.labels:
                        have.labels.append(lab)
                if _status_rank(r.status) > _status_rank(have.status):
                    have.status = r.status          # termination (ended/retracted) wins
                if not have.valid_to and r.valid_to:
                    have.valid_to = r.valid_to      # fill an unknown bound from the duplicate
            else:
                by_pair[kk] = r
                self.relations.append(r)
        # facts are per-occurrence: dedupe only EXACT repeats (e.g. reflexion re-stating
        # something the first pass already found), never distinct occurrences.
        factset = {f.dedup_key() for f in self.facts}
        for f in other.facts:
            if f.dedup_key() not in factset:
                self.facts.append(f); factset.add(f.dedup_key())
        return self


class Extractor(Protocol):
    name: str

    def extract_text(self, text: str, title: str = "") -> Extraction: ...
    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction: ...
    def extract_url(self, url: str) -> Extraction: ...
    def extract_commit(self, message: str, diff: str) -> Extraction: ...
    def extract_repo(self, signals: dict) -> Extraction: ...


# --------------------------------------------------------------------------- #
# Structured-output tool schema (shared by the LLM path)
# --------------------------------------------------------------------------- #
_ENTITY_ENUM = [t.value for t in EntityType]
_CATEGORY_ENUM = [c.value for c in EntityCategory]

GRAPH_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_graph",
        "description": "Emit the knowledge-graph elements extracted from the content.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": _ENTITY_ENUM},
                            "category": {
                                "type": "string", "enum": _CATEGORY_ENUM,
                                "description": "Broad glyph category for display: 'person' "
                                "for people and named characters, 'place' for geographic "
                                "locations, venues, rooms, landmarks and buildings, 'thing' "
                                "for everything else (orgs, objects, products, works, events, "
                                "named abstractions).",
                            },
                        },
                        "required": ["name", "type", "category"],
                    },
                },
                "tags": {"type": "array", "items": {"type": "string"},
                         "description": "5-12 lowercase topical tags."},
                "relations": {
                    "type": "array",
                    "description": "Directed relationships between entities.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "labels": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "1-3 short lowercase relationship labels "
                                "read SOURCE→TARGET, e.g. 'founded', 'works_with', "
                                "'member_of', 'located_in', 'parent_of'. Use the BASE "
                                "predicate even for past relationships (use 'works_with' + "
                                "status 'ended', NOT 'former_colleague'). Consolidated "
                                "automatically.",
                            },
                            "status": {
                                "type": "string", "enum": ["asserted", "ended", "retracted"],
                                "description": "'asserted' if the relationship holds; 'ended' "
                                "if the text says it TERMINATED (former, ex-, no longer, left, "
                                "until X) — it WAS true and stopped; 'retracted' if the text "
                                "CORRECTS a prior claim as mistaken — it was NEVER true "
                                "('actually X never...', 'I was wrong that...'). Default "
                                "'asserted'.",
                            },
                            "valid_from": {"type": "string",
                                           "description": "optional ISO date/year the fact began, if stated"},
                            "valid_to": {"type": "string",
                                         "description": "optional ISO date/year the fact ended, if stated"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["source", "target", "labels"],
                    },
                },
                "facts": {
                    "type": "array",
                    "description": "EVERY stated amount/count/measurement, always — "
                    "exempt from the entities[] salience filter. subject: who/what it "
                    "belongs to. predicate: short verb (earned/spent/paid/weighed/ran). "
                    "value: bare number. unit: currency/unit, else empty. date: ISO "
                    "date if stated, else empty.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "value": {"type": "number"},
                            "unit": {"type": "string"},
                            "date": {"type": "string"},
                        },
                        "required": ["subject", "predicate", "value"],
                    },
                },
                "description": {"type": "string",
                                "description": "One-line description (images only)."},
            },
            "required": ["entities", "tags"],
        },
    },
}


# Prefixes that fold a relationship label onto its base predicate + a termination flag.
# These are the UNAMBIGUOUS former-markers: whatever follows is closed unconditionally.
_TERM_PREFIX = re.compile(r"^(former|formerly|ex|no[\s_-]?longer|used[\s_-]?to)[\s_-]+",
                          re.I)

# 'past'/'once' are AMBIGUOUS: "once_met"/"once_lived_in" read as "at one time it HAPPENED"
# and "past_project"/"past_month" name a thing/time-period, none of which are terminations —
# folding them unconditionally wrongly CLOSED live facts, which is why they were dropped.
# They are re-admitted here but GUARDED: they only fold when the remaining label is a
# recognized relation predicate (`_KNOWN_PREDICATES`). So "past_employer" -> employer+ended
# (mergeable with an open 'employer' fact), while "past_month"/"past_project"/"once_met"
# keep an unrecognized remainder and flow through as 'asserted', unchanged from before.
_GUARDED_TERM_PREFIX = re.compile(r"^(past|once)[\s_-]+", re.I)

# Relation predicates a 'past'/'once' marker may legitimately close. Seeded from the frozen
# predicate hint in _SYS, plus the relationship/role nouns that commonly wear a former-marker.
_KNOWN_PREDICATES = frozenset({
    "founded", "founded_by", "works_with", "member_of", "located_in", "part_of",
    "parent_of", "child_of", "created", "created_by", "discovered", "employed_by",
    "succeeded_by", "influenced_by",
    "employer", "employee", "colleague", "coworker", "co_worker", "boss", "manager",
    "partner", "spouse", "husband", "wife", "friend", "mentor", "student", "teacher",
    "member", "member_of_staff", "works_at", "works_for", "reports_to", "resident_of",
})


def _strip_term_prefix(label: str) -> tuple[str, bool]:
    """Return (base_predicate, ended) for a single label. An unambiguous former-marker
    strips unconditionally; a guarded 'past'/'once' marker strips ONLY when the remainder
    is a recognized relation predicate, leaving documented false positives untouched."""
    base = _TERM_PREFIX.sub("", label)
    if base != label:
        return base, True
    guarded = _GUARDED_TERM_PREFIX.sub("", label)
    if guarded != label and guarded.lower() in _KNOWN_PREDICATES:
        return guarded, True
    return label, False


def _coerce_labels(r: dict) -> list[str]:
    """Open-vocab labels[], with back-compat for an old single `relation` string."""
    labels = [str(x).strip() for x in (r.get("labels") or []) if str(x).strip()]
    if not labels and r.get("relation"):
        labels = [str(r["relation"]).strip()]
    # de-dupe preserving order
    return list(dict.fromkeys(labels))


def _normalize_termination(labels: list[str]) -> tuple[list[str], bool]:
    """Fold tense/aspect wrappers onto the base predicate + a termination flag
    (docs/TEMPORAL.md §7): former_colleague / ex-coworker / no_longer_works_with →
    base predicate + ended=True, so the temporal layer CLOSES the base fact rather than
    minting an `ex_*` predicate that would sprawl the vocabulary."""
    ended = False
    out = []
    for lab in labels:
        stripped = lab.strip()
        base, is_term = _strip_term_prefix(stripped)
        if is_term:
            ended = True
        out.append(base or stripped)
    return list(dict.fromkeys(l for l in out if l)), ended


def _resolve_status(r: dict, ended: bool) -> str:
    """Map a relation's raw `status` (+ a termination-prefix flag) onto the three
    polarities the temporal layer understands: `asserted` (holds), `ended` (was true, then
    stopped → valid-time close) or `retracted` (a correction: never actually true →
    belief flip). `retracted` wins over a stray termination prefix."""
    raw = str(r.get("status", "")).lower()
    if raw == "retracted":
        return "retracted"
    if ended or raw == "ended":
        return "ended"
    return "asserted"


# Entity-type aliases the parse path accepts beyond the EntityType enum values: the fork
# authored named abstractions as type 'term' (its enum had no CONCEPT), and payloads from
# it must keep rebuilding here. Unknown types still fall to OTHER.
_ENTITY_TYPE_ALIASES = {"term": EntityType.CONCEPT}


_WEEKDAY_RE = re.compile(
    r"^(mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)(?:day)?s?$", re.I)
_NUMERIC_RE = re.compile(r"^\$?[\d][\d,.]*%?$")
_COMPOUND_SPLIT_RE = re.compile(r"\s*,\s*|\s+and\s+", re.I)


def _split_compound(value: str) -> list[str]:
    """Split an object/subject string naming a LIST of distinct items ("Tuesdays and
    Thursdays", "$200 and $50") into its parts, so a single compound mention doesn't
    flatten N true occurrences into 1 edge/fact. Conservative on purpose: only splits
    when every resulting part is short, DISTINCT, and looks like a list item (a weekday
    or a number) — this is what stops "Bed and Breakfast" or "Johnson and Johnson" (a
    single proper name that happens to contain "and") from being wrongly split."""
    parts = [p.strip() for p in _COMPOUND_SPLIT_RE.split(value.strip()) if p.strip()]
    if len(parts) < 2:
        return [value]
    if len(parts) != len({p.lower() for p in parts}):
        return [value]           # duplicate parts → a proper name ("Johnson and Johnson")
    if all(len(p.split()) <= 2 and (_WEEKDAY_RE.match(p) or _NUMERIC_RE.match(p))
           for p in parts):
        return parts
    return [value]


def _parse_tool_payload(payload: dict) -> Extraction:
    ents = []
    for e in payload.get("entities", []) or []:
        name = (e.get("name") or "").strip()
        if not name:
            continue
        raw_type = str(e.get("type", "other")).strip().lower()
        try:
            etype = _ENTITY_TYPE_ALIASES.get(raw_type) or EntityType(raw_type)
        except ValueError:
            etype = EntityType.OTHER
        # Optional glyph category from the payload; anything unstated/unknown falls back
        # to the type-derived category so ExtractedEntity.category is always populated.
        try:
            category = EntityCategory(str(e.get("category", "") or "").strip().lower())
        except ValueError:
            category = entity_category_for_type(etype)
        ents.append(ExtractedEntity(name=name, type=etype, category=category))
    # back-compat: accept the "tags" key, falling back to the fork's "concepts" key so
    # extractions authored against the fork's schema still rebuild. Process/meta tags in
    # the stoplist are dropped — they are about the act of note-taking, not the content.
    tags = [surface for t in (payload.get("tags") or payload.get("concepts") or [])
            for surface in [str(t).strip() if t else ""]
            if surface and surface.lower() not in TAG_STOPLIST]
    rels = []
    for r in payload.get("relations", []) or []:
        s, t = (r.get("source") or "").strip(), (r.get("target") or "").strip()
        labels, ended = _normalize_termination(_coerce_labels(r))
        if not s or not t or not labels:
            continue
        status = _resolve_status(r, ended)
        for target_part in _split_compound(t):
            rels.append(ExtractedRelation(
                source=s, target=target_part, labels=labels,
                provenance=Provenance.EXTRACTED,
                confidence=float(r.get("confidence", 0.8)),
                status=status, valid_from=str(r.get("valid_from", "") or ""),
                valid_to=str(r.get("valid_to", "") or ""),
            ))
    facts = []
    for f in payload.get("facts", []) or []:
        subj = (f.get("subject") or "").strip()
        pred = (f.get("predicate") or "").strip()
        if not subj or not pred or f.get("value") is None:
            continue
        try:
            value = float(f["value"])
        except (TypeError, ValueError):
            continue
        unit = str(f.get("unit", "") or "").strip()
        date = str(f.get("date", "") or "").strip()
        for subj_part in _split_compound(subj):
            facts.append(ExtractedFact(subject=subj_part, predicate=pred, value=value,
                                       unit=unit, date=date))
    return Extraction(entities=ents, tags=tags, relations=rels, facts=facts,
                      description=(payload.get("description") or None))


# --------------------------------------------------------------------------- #
# URL ingest: fetch page signals + subject-scoped extraction prompt
# --------------------------------------------------------------------------- #
# kg.ingest owns an identical _HTML_TITLE, but importing it here would be circular
# (ingest imports extractors), so the same tiny pattern is redefined locally.
_HTML_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_RE = re.compile(
    r"<meta\s+[^>]*?(?:property|name)\s*=\s*[\"']([^\"']+)[\"'][^>]*?"
    r"content\s*=\s*[\"']([^\"']*)[\"']", re.IGNORECASE | re.DOTALL)
_META_RE_REV = re.compile(  # content=... before property/name (attribute order varies)
    r"<meta\s+[^>]*?content\s*=\s*[\"']([^\"']*)[\"'][^>]*?"
    r"(?:property|name)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE | re.DOTALL)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


class _BodyTextParser(HTMLParser):
    """Stdlib fallback reader: collect visible text, skipping script/style/nav chrome.
    Used only when trafilatura is unavailable (fail-soft, PROTOCOL boilerplate-strip)."""
    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self._parts)


def _strip_tags(raw: str | None) -> str:
    return html.unescape(_TAG_RE.sub(" ", raw or "")).strip() if raw else ""


def _main_text(html_doc: str) -> str:
    """Main body text with boilerplate stripped. trafilatura when available; else a stdlib
    html.parser reader; else the tag-stripped raw body — never raises."""
    try:
        import trafilatura
        got = trafilatura.extract(html_doc, include_comments=False, include_tables=False)
        if got and got.strip():
            return got.strip()
    except Exception:  # noqa: BLE001 — missing dep or a parse quirk must fail soft
        pass
    try:
        p = _BodyTextParser()
        p.feed(html_doc)
        got = p.text()
        if got.strip():
            return got.strip()
    except Exception:  # noqa: BLE001
        pass
    return _strip_tags(html_doc)


def fetch_page(url: str, timeout: float = 4.0, max_bytes: int = 2_000_000) -> dict:
    """Fetch a URL and return its structural signals + main body text:
    {url, domain, title, og_title, meta_description, h1, body}. Fail-soft — any network
    or parse problem yields just {url, domain}. Chunked reads under a wall-clock deadline
    (mirrors kg/ingest.py:_resolve_link_title) so a tarpit can't hang ingest.
    KG_LINK_FETCH=0 disables the network fetch entirely (hermetic/offline runs)."""
    domain = urllib.parse.urlparse(url).netloc
    signals = {"url": url, "domain": domain, "title": "", "og_title": "",
               "meta_description": "", "h1": "", "body": ""}
    if os.environ.get("KG_LINK_FETCH", "1") == "0":
        return signals
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "you-kg/0.1"})
        deadline = time.monotonic() + 2 * timeout
        raw = b""
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while len(raw) < max_bytes and time.monotonic() < deadline:
                chunk = resp.read(65536)
                if not chunk:
                    break
                raw += chunk
        doc = raw.decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001 — a dead link must never fail ingest
        return signals
    metas: dict[str, str] = {}
    for pat, order in ((_META_RE, "kv"), (_META_RE_REV, "vk")):
        for a, b in pat.findall(doc):
            key, val = (a, b) if order == "kv" else (b, a)
            metas.setdefault(key.strip().lower(), _strip_tags(val))
    tm = _HTML_TITLE.search(doc)
    h1m = _H1_RE.search(doc)
    signals["title"] = _strip_tags(tm.group(1)) if tm else ""
    signals["og_title"] = metas.get("og:title", "")
    signals["meta_description"] = metas.get("description") or metas.get("og:description", "")
    signals["h1"] = _strip_tags(h1m.group(1)) if h1m else ""
    signals["body"] = _main_text(doc)
    return signals


_URL_INSTRUCTION = (
    "You are extracting a knowledge-graph record for a saved web page. You are given the\n"
    "page's URL/domain, title, og:title, meta description, headings, and main body text.\n\n"
    "Capture WHAT THIS PAGE IS ABOUT — its primary subject — not an inventory of everything\n"
    "named on it.\n\n"
    "1. Identify the SINGLE primary subject: the entity, product, organization, or topic the\n"
    "   page is fundamentally about. The domain, title, og:title, and H1 are the strongest\n"
    "   signals; the body confirms. For a company's or product's own site, the subject is that\n"
    "   company/product.\n\n"
    "2. Write `description`: ONE specific, self-contained sentence stating what the primary\n"
    "   subject is. This is the page's only retrieval surface — a reader who never sees the\n"
    "   page must understand what it is from this line alone.\n\n"
    "3. ENTITIES. Emit the primary subject as an entity, then the subject's salient DOMAIN\n"
    "   CONCEPTS, topics, and features as `concept`-typed entities (category 'thing') — e.g.\n"
    "   'corporate card', 'spend management', 'finance automation', 'receipt capture'. Type\n"
    "   these exactly as the rest of the graph types ideas / fields / methods / categories, so\n"
    "   the same concept RESOLVES to one shared node when it recurs on another page. Use short\n"
    "   canonical names ('corporate card', not 'unlimited virtual + physical corporate cards').\n"
    "   Be reasonably COMPLETE about the subject: what it is, its category, what it does, its\n"
    "   features / products / offerings, and where it operates.\n\n"
    "4. TAGS. Emit 5-12 lowercase topical themes describing what the page is about.\n\n"
    "5. RELATIONS. Connect the primary subject to the concepts / entities above. EVERY relation\n"
    "   endpoint MUST be a `name` you already listed under entities or tags — never coin a new\n"
    "   phrase as a relation target (that mints junk nodes). Put stated amounts / counts /\n"
    "   measurements in `facts`, not relations.\n\n"
    "EXCLUDE noise that is NOT about the primary subject — do NOT extract:\n"
    "- OTHER entities merely mentioned in passing: partners, competitors, banks/brands\n"
    "  name-dropped, examples, comparisons.\n"
    "- Cross-links, \"related articles\", \"you may also like\", recommended/related products.\n"
    "- Navigation, legal/boilerplate, cookie/privacy notices, author bios, ads.\n\n"
    "A DIFFERENT, secondary entity may be included ONLY if the page states a CONCRETE\n"
    "relationship between it and the primary subject (e.g. \"CardCo is issued by BankX\") —\n"
    "then emit it as a relation with the primary subject as one endpoint. If it is merely\n"
    "named nearby, drop it.\n\n"
    "Stay ON the primary subject, but don't be stingy about it: a record that omits what the\n"
    "page actually says about its own subject is too lean. Only when a detail clearly belongs\n"
    "to something OTHER than the primary subject, leave it out.\n"
    "Emit exactly one emit_graph call."
)


def _assemble_url_block(signals: dict, body_cap: int) -> str:
    """One text block: the page signals + the subject-scoped extraction instruction."""
    body = (signals.get("body") or "")[:body_cap]
    lines = [
        f"URL: {signals.get('url', '')}",
        f"Domain: {signals.get('domain', '')}",
        f"Title: {signals.get('title', '')}",
        f"og:title: {signals.get('og_title', '')}",
        f"Meta description: {signals.get('meta_description', '')}",
        f"H1: {signals.get('h1', '')}",
        "",
        "Main body text:",
        body or "(no body text could be extracted)",
        "",
        _URL_INSTRUCTION,
    ]
    return "\n".join(lines)


def _fallback_url_description(signals: dict) -> str:
    """A description surface when no LLM authored one (offline/local floor): best available
    page signal, else the bare domain."""
    for key in ("meta_description", "og_title", "title", "h1"):
        val = (signals.get(key) or "").strip()
        if val:
            return val
    return f"A web page at {signals.get('domain') or signals.get('url') or 'an unknown URL'}."


# --------------------------------------------------------------------------- #
# Image ingest: MIME sniff + graph-aligned vision prompt
# --------------------------------------------------------------------------- #
# Sniff the real image type for the data URI: a PNG/webp/gif sent as image/jpeg can be
# rejected or mis-decoded by the vision endpoint.
_IMAGE_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".webp": "image/webp", ".gif": "image/gif"}
# The image formats the vision path can actually serve (every provider — OpenAI/Anthropic
# APIs and the codex CLI — accepts exactly these). The engine's fast-fail attachment
# validation imports this so its answer can't drift from the extractor's.
SUPPORTED_IMAGE_EXTS = frozenset(_IMAGE_MIME)

# Content magic numbers: an import can hand us bytes in an extension-less tempfile
# (kg/imports/normalize.py writes ".img"), so the header settles what the bytes are
# when the extension doesn't.
_IMAGE_MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"),
                (b"GIF87a", "image/gif"), (b"GIF89a", "image/gif"))


def _sniff_image_mime(path: str) -> str:
    """The image's real MIME type, from the extension or (for unknown extensions) the file
    header. A format outside _IMAGE_MIME raises UnsupportedMedia — sending it anyway (the
    old default-to-jpeg) makes every provider fail or, worse, answer 'I can't see an image'
    and have that junk persisted as the episode's retrieval surface."""
    ext = os.path.splitext(path or "")[1].lower()
    mime = _IMAGE_MIME.get(ext)
    if mime:
        return mime
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        head = b""
    for magic, m in _IMAGE_MAGIC:
        if head.startswith(magic):
            return m
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    raise UnsupportedMedia(
        f"can't process media format {ext or '(no extension)'!r} — "
        "supported image formats: JPEG, PNG, WebP, GIF")


# Graph-aligned so an ingested image lands as the SAME shape of record a conversation
# produces — abstracted to shared concept/work/person/place/org nodes and me-anchored
# relations — never a catalog of literal visual objects (bicycle/trees/sky).
_IMAGE_INSTRUCTION = (
    "You are extracting a knowledge-graph record for an image the user saved to their personal\n"
    "memory. The image is provided directly; a caption/note the user wrote may accompany it.\n\n"
    "First decide the kind of image:\n"
    "- TEXT-HEAVY (screenshot, receipt, slide, tweet, chat, document, sign): the value is the\n"
    "  TEXT. Read it, and put its meaningful content into the record — the entities, amounts,\n"
    "  names, and claims it states — exactly as if that text had been typed as a note.\n"
    "- PHOTOGRAPHIC (a scene, object, place, person, meal, whiteboard): describe what is depicted\n"
    "  and identify everything meaningful in it.\n\n"
    "1. `description`: ONE specific, self-contained sentence saying what the image is/shows. This\n"
    "   is the image's retrieval surface — a reader who never sees it must understand it from\n"
    "   this line alone.\n\n"
    "2. ENTITIES — extract everything identifiable: named things, objects, people, places,\n"
    "   products. Use the graph's types: person, place, org, work (a named product / brand /\n"
    "   creative work / food / item), concept (an activity, topic, field, or idea), event, date.\n\n"
    "   CRITICAL — connective tissue: alongside the concrete things, you MUST also emit the\n"
    "   abstract CONCEPTS and topics the image is about (a bike on a trail → also `cycling` /\n"
    "   `commuter bike`; a plate of food → also the cuisine/dish concept). These shared concepts\n"
    "   are what connect this image to the user's notes about the same things — an image with\n"
    "   only concrete visual objects and no concepts becomes an isolated island. Concrete detail\n"
    "   is welcome; concepts are required.\n\n"
    "   Use short canonical names ('Greek yogurt', not 'a plastic tub of yogurt on a table') so a\n"
    "   thing RESOLVES to one shared node when it recurs elsewhere.\n\n"
    "3. TAGS — 4-10 lowercase topical themes describing what the image is about.\n\n"
    "4. RELATIONS — connect the subjects to their concepts/entities. EVERY relation endpoint\n"
    "   MUST be a name you already listed under entities or tags — never coin a new phrase as a\n"
    "   relation target (that mints junk nodes). Put stated amounts / counts / prices in `facts`.\n\n"
    "   FIRST-PERSON `me` relations REQUIRE EVIDENCE, never assumption. Anchor something to\n"
    "   `me` (me --rode--> commuter bike; me --ate--> ramen; me --visited--> Reykjavik) ONLY when\n"
    "   the caption says so in the user's own words, OR the image itself self-evidently shows the\n"
    "   user's OWN action or possession — a selfie, a meal they are eating, an object held/worn/\n"
    "   owned, a first-person point-of-view shot. A photo that merely DEPICTS a place, landmark,\n"
    "   object, meal, or scene is NOT evidence the user was there or did anything with it: a saved\n"
    "   image may be a reference, an inspiration, a screenshot, or someone else's photo. When in\n"
    "   doubt, emit NO `me` relation — capture what the image is ABOUT (its entities, concepts,\n"
    "   and description, which are what make it findable later) and stop there.\n\n"
    "When a caption is provided, treat the user's own words as the strongest signal of what\n"
    "matters — including whether a first-person relation is warranted — and align the visual\n"
    "extraction to it.\n\n"
    "Emit exactly one emit_graph call."
)


# --------------------------------------------------------------------------- #
# Code ingest: git commit + repo-summary extraction prompts (kg/code/)
# --------------------------------------------------------------------------- #
# A single git commit the user authored → an immutable Episode of their own work. The
# message states intent (often vague); the diff is ground truth — use BOTH and union them.
_COMMIT_INSTRUCTION = (
    "You are extracting a knowledge-graph record for a single git commit the user authored, as\n"
    "part of their personal memory of their own work. You are given the commit MESSAGE and DIFF.\n\n"
    "Use BOTH. The message states intent but is often incomplete or vague (\"fix bug\", \"wip\"); the\n"
    "diff is the ground truth of what actually changed. Extract what the commit did from the\n"
    "message AND what the diff shows, and union them — never rely on the message alone.\n\n"
    "1. `description`: ONE specific, first-person sentence saying what the user did in this commit\n"
    "   (\"Fixed a double-charge bug in webhook retry by de-duplicating on idempotency key.\"). This\n"
    "   is the commit's retrieval surface.\n\n"
    "2. ENTITIES — the graph's normal types (person, org, work, concept). Emit:\n"
    "   - the code COMPONENTS changed (files/modules/functions/classes) as work/concept entities,\n"
    "     short qualified names ('webhook retry handler', 'payments.py');\n"
    "   - the DOMAIN CONCEPTS involved ('idempotency', 'rate limiting', 'authentication') —\n"
    "     CRITICAL connective tissue: these link this commit to the user's notes, conversations,\n"
    "     and other projects about the same ideas;\n"
    "   - any TECH/libraries the change touches.\n\n"
    "3. TAGS — 3-8 lowercase themes ('bugfix', 'payments', 'webhooks', 'refactor').\n\n"
    "4. RELATIONS — this is the user's own work, so anchor to `me`:\n"
    "   me --fixed / added / refactored / removed--> <component or concept>, and connect\n"
    "   components to concepts (webhook handler --handles--> idempotency). EVERY relation endpoint\n"
    "   MUST be a name you listed under entities or tags — never coin a new relation target.\n\n"
    "Ignore pure formatting, whitespace, lockfile churn, and version bumps. Emit exactly one\n"
    "emit_graph call."
)

_REPO_INSTRUCTION = (
    "You are extracting a knowledge-graph record for a code project the user has, as part of their\n"
    "personal memory. You are given the project name, README, dependency manifests, and a summary\n"
    "of its file/directory structure.\n\n"
    "1. `description`: ONE specific sentence saying what this project IS and DOES (\"A personal-\n"
    "   finance app that syncs bank transactions and auto-categorizes spending.\"). Retrieval surface.\n\n"
    "2. ENTITIES:\n"
    "   - the PROJECT itself (work/project entity, short name);\n"
    "   - its DOMAIN CONCEPTS — what it does ('transaction sync', 'spend categorization') —\n"
    "     CRITICAL connective tissue to the user's notes and conversations;\n"
    "   - its TECH STACK — languages, frameworks, libraries (from the manifests), as entities.\n\n"
    "3. TAGS — 4-10 lowercase themes.\n\n"
    "4. RELATIONS — first-person: me --built / works_on--> project; project --uses--> <library>;\n"
    "   project --does / implements--> <concept>. EVERY endpoint MUST be a listed entity or tag.\n\n"
    "Stay on what the project is and does; don't inventory every file. Emit exactly one emit_graph call."
)


def _assemble_commit_block(message: str, diff: str, diff_cap: int) -> str:
    """One text block: the commit message + (capped) diff + the commit extraction instruction."""
    return "\n".join([
        "Commit message:",
        (message or "").strip() or "(empty)",
        "",
        "Diff:",
        (diff or "")[:diff_cap] or "(no source diff)",
        "",
        _COMMIT_INSTRUCTION,
    ])


def _assemble_repo_block(signals: dict, cap: int) -> str:
    """One text block: the repo signals (name, README, manifests, tree) + the repo instruction.
    Mirrors _assemble_url_block — a deterministic gather feeding a subject-scoped prompt."""
    manifests = signals.get("manifests") or {}
    libs = signals.get("libraries") or []
    parts = [
        f"Project name: {signals.get('name', '')}",
        "",
        "README:",
        (signals.get("readme") or "")[:cap] or "(no README)",
        "",
        "Dependency manifests:",
    ]
    for fname, body in manifests.items():
        parts.append(f"--- {fname} ---")
        parts.append((body or "")[:2000])
    if libs:
        parts.append("")
        parts.append("Declared dependencies (parsed from manifests): " + ", ".join(libs))
    parts += [
        "",
        "File/directory structure:",
        (signals.get("tree") or "")[:4000] or "(unavailable)",
        "",
        _REPO_INSTRUCTION,
    ]
    return "\n".join(parts)


def _concise_commit_description(message: str) -> str:
    """Offline/local floor description surface for a commit: its subject line."""
    subject = (message or "").strip().splitlines()[0].strip() if (message or "").strip() else ""
    return subject or "A git commit."


def _fallback_repo_description(signals: dict) -> str:
    """Offline/local floor description surface for a repo: the README's first line, else name."""
    readme = (signals.get("readme") or "").strip()
    if readme:
        first = readme.splitlines()[0].lstrip("# ").strip()
        if first:
            return first
    name = signals.get("name") or "a code project"
    return f"A code project named {name}."


# --------------------------------------------------------------------------- #
# OpenAI (real)
# --------------------------------------------------------------------------- #
class OpenAIExtractor:
    name = "llm"

    # The system prompt + GRAPH_TOOL are kept STATIC and BLIND (no live graph
    # vocabulary is ever injected) — extraction never diffs against existing state;
    # that is a separate downstream concern (canonicalize.py + the L3 tie-breaker).
    # The predicate list below is a FROZEN soft hint, not the growing vocabulary.
    _SYS = (
        "You extract a knowledge graph from a single piece of content. Work in this order.\n\n"
        "1) ENTITIES. List the salient, nameable entities, each with a type:\n"
        "   - person  — an individual human (e.g. Marie Curie)\n"
        "   - place   — a geographic location (e.g. Paris, the Pacific Ocean)\n"
        "   - org     — an organisation, company, institution, team, or group\n"
        "   - concept — an idea, field, method, material, or abstract thing (e.g. radioactivity)\n"
        "   - work    — a named created work (book, film, song, paper, artwork, product)\n"
        "   - event   — a time-bounded happening (a war, election, discovery, ceremony)\n"
        "   - other   — a real entity that fits none of the above\n"
        "   Also give each entity a CATEGORY for the graph's glyphs: 'person' for people and "
        "named characters, 'place' for geographic locations, venues, rooms, landmarks and "
        "buildings, 'thing' for everything else (organisations, objects, products, works, "
        "events, named abstractions).\n"
        "   Prefer the fullest proper name the content uses (\"John F. Kennedy\", not \"JFK\"). "
        "Do not invent entities not in the content. A handful is fine; do not pad.\n\n"
        "2) TAGS. Emit 5-12 lowercase topical tags describing what the content is ABOUT "
        "(themes, not entities).\n\n"
        "3) RELATIONS. Emit the key DIRECTED relationships. RULES:\n"
        "   - Source and target are normally entities from step 1 — use the EXACT SAME "
        "surface string you wrote there. You MAY also use one of your step-2 tags as an "
        "endpoint when the note states a relationship to a topic or theme (e.g. an org "
        "'works_on' a field). Never relate something you named as neither an entity nor a tag.\n"
        "   - Each relationship has 1-3 short lowercase labels that read SOURCE then TARGET. "
        "Order matters. Example: Marie Curie discovered polonium → source \"Marie Curie\", "
        "target \"polonium\", label \"discovered\" (NOT the reverse).\n"
        "   - Keep voice as written: \"X founded by Y\" may be source \"Y\" target \"X\" label "
        "\"founded\", OR source \"X\" target \"Y\" label \"founded_by\" — but never silently flip "
        "a label's voice. \"founded\" and \"founded_by\" are different and both are fine.\n"
        "   - For a mutually symmetric relationship (works_with, married_to, sibling_of) emit "
        "it once, in one direction only.\n"
        "   - Use natural predicate names. Prefer one of these common forms when it genuinely "
        "fits, otherwise coin your own short lowercase predicate — this list is a HINT, NOT a "
        "fixed vocabulary: founded, founded_by, works_with, member_of, located_in, part_of, "
        "parent_of, child_of, created, created_by, discovered, employed_by, succeeded_by, "
        "influenced_by. Similar labels are consolidated automatically downstream, so do not "
        "try to match any canonical form yourself.\n"
        "   - TIME. If the text says a relationship ENDED (former, ex-, no longer, left, "
        "until X), emit the BASE predicate with status 'ended' — never coin 'former_*' or "
        "'ex_*'. If the text CORRECTS a prior claim as having been mistaken all along ('I "
        "was wrong that…', 'actually X never…'), use status 'retracted' instead (it was "
        "never true, vs 'ended' which was true and stopped). Set valid_from/valid_to ONLY "
        "when the text states a date; otherwise leave them empty (it defaults to 'as of this "
        "content'). Do not guess dates.\n"
        "   - Few or no relations is fine if the content doesn't clearly state them.\n\n"
        "4) FACTS. Extract EVERY stated amount/count/measurement into facts[], no "
        "salience filter. One per OCCURRENCE — a repeat (2nd purchase/visit/class) or "
        "list (\"$200 and $50\") is 2 facts, not 1.\n\n"
        "Call emit_graph exactly once."
    )

    # Appended to _SYS ONLY when config.self_entity is on (personal-web mode). The
    # narrator is named exactly 'me' so it canonicalizes onto the single self anchor.
    _FIRST_PERSON_CLAUSE = (
        "\n\nFIRST PERSON: if the content is narrated in the first person (I/me/my), "
        "include the narrator as an entity named exactly 'me' (type person), and use 'me' "
        "as the source or target of any relationship the narrator participates in (e.g. "
        "source 'me', target 'Becky', label 'had_coffee_with'). Do not invent a name for "
        "the narrator."
    )

    def __init__(self, config: Config):
        self.config = config
        self.client = make_client()  # active provider (explicit KG_LLM, else auto-detected)
        self.meter = UsageMeter()

    @property
    def _system(self) -> str:
        """Effective system prompt. OFF-path returns _SYS BYTE-FOR-BYTE; personal-web mode
        appends the first-person clause so the base prompt never drifts when the feature
        is off."""
        if self.config.self_entity:
            return self._SYS + self._FIRST_PERSON_CLAUSE
        return self._SYS

    def _call(self, content_blocks: list) -> Extraction:
        import json
        # explicit llm_model wins; unset resolves to the active provider's default
        model = resolve_model(self.config.llm_model)
        with prof_span("extract.llm"):
            msg = call_with_backoff(lambda: self.client.chat.completions.create(
                model=model,
                max_tokens=self.config.extract_max_tokens,
                temperature=0,
                messages=[
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": content_blocks},
                ],
                tools=[GRAPH_TOOL],
                tool_choice={"type": "function", "function": {"name": "emit_graph"}},
            ))
        self.meter.record("extract", model or "cli-default", msg)
        tc = getattr(msg.choices[0].message, "tool_calls", None) if msg.choices else None
        if tc and tc[0].function.name == "emit_graph":
            return _parse_tool_payload(json.loads(tc[0].function.arguments))
        return Extraction()

    def _reflexion(self, text: str, first: Extraction) -> Extraction:
        if not self.config.reflexion:
            return first
        tags = ", ".join(first.tags) or "(none)"
        ents = ", ".join(e.name for e in first.entities) or "(none)"
        facts_found = ", ".join(
            f"{f.subject} {f.predicate} {f.value}{f.unit}" for f in first.facts) or "(none)"
        prompt = (
            f"Content:\n{text[:4000]}\n\n"
            f"First pass found these tags: {tags}\n"
            f"and these entities: {ents}\n"
            f"and these facts (amounts/counts/measurements): {facts_found}.\n\n"
            "Now do a focused recall check. List ONLY items you OMITTED:\n"
            "- any salient entity in the content missing above (with its type and category),\n"
            "- any important topical tag not already emitted,\n"
            "- any clearly-stated directed relationship between entities you missed "
            "(source and target must be named entities; labels read source→target),\n"
            "- any amount/count/measurement missing from the facts list above (not "
            "salience-filtered — emit it), including a collapsed repeat occurrence or "
            "multi-value mention (\"$200 and $50\" is two facts).\n\n"
            "Do not repeat anything already found. If you omitted nothing, return empty "
            "arrays. Call emit_graph exactly once with only the missed items."
        )
        try:
            extra = self._call([{"type": "text", "text": prompt}])
            return first.merge(extra)
        except Exception:
            return first

    def extract_text(self, text: str, title: str = "") -> Extraction:
        header = f"Title: {title}\n\n" if title else ""
        # Per-call input cap (config-driven). Sectioning keeps each slice <= long_doc_chars,
        # and extract_max_chars >= long_doc_chars, so this never truncates a section. It only
        # bites a single un-sectioned call whose text exceeds the cap.
        first = self._call([{"type": "text", "text": header + text[:self.config.extract_max_chars]}])
        return self._reflexion(text, first)

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        """Perceive an image into a graph-aligned record (the SAME shape a conversation
        produces): a one-line `description` retrieval surface plus concept/work/person/place
        entities and me-anchored relations — never a catalog of literal visual objects. The
        data URI's MIME is sniffed from the real extension / file header (a PNG sent as
        image/jpeg fails); an unsupported format raises UnsupportedMedia before any read."""
        mime = _sniff_image_mime(image_path)
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        hint = f"\n\nThe image may contain: {label_hint}." if label_hint else ""
        blocks = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            {"type": "text", "text": _IMAGE_INSTRUCTION + hint},
        ]
        ext = self._call(blocks)
        if not ext.description:
            ext.description = (label_hint and f"A photo containing {label_hint}.") or "An image."
        return ext

    def extract_url(self, url: str) -> Extraction:
        """Fetch a saved URL and extract a SUBJECT-SCOPED graph record: the description
        (the LINK episode's embedding surface) names the page's primary subject, and only
        that subject's own tags/entities/relations are kept (strict — no passing mentions).
        The full fetched body rides along on ext.source_text for the SOURCE provenance node."""
        signals = fetch_page(url)
        ext = self._call([{"type": "text",
                           "text": _assemble_url_block(signals, self.config.extract_max_chars)}])
        if not ext.description:
            ext.description = _fallback_url_description(signals)
        ext.source_text = signals.get("body") or None
        ext.page_title = signals.get("title") or signals.get("og_title") or None
        return ext

    def extract_commit(self, message: str, diff: str) -> Extraction:
        """Extract a first-person memory record for one git commit from its MESSAGE + DIFF
        (union of intent and ground truth). The description is the retrieval surface; the
        full diff rides along on ext.source_text for the un-rankable SOURCE provenance node
        (same pattern as extract_url), so the raw change stays auditable without embedding it."""
        ext = self._call([{"type": "text",
                            "text": _assemble_commit_block(message, diff,
                                                           self.config.extract_max_chars)}])
        if not ext.description:
            ext.description = _concise_commit_description(message)
        ext.source_text = diff or None
        return ext

    def extract_repo(self, signals: dict) -> Extraction:
        """Extract a repo-summary record (mirrors extract_url): the description names what the
        project is and does; entities pre-populated from manifests + domain concepts the LLM adds."""
        ext = self._call([{"type": "text",
                            "text": _assemble_repo_block(signals, self.config.extract_max_chars)}])
        if not ext.description:
            ext.description = _fallback_repo_description(signals)
        return ext


# --------------------------------------------------------------------------- #
# Shared text-extraction helper (sectioning for long docs, §9 risk 4)
# --------------------------------------------------------------------------- #
def extract_text_sectioned(extractor: Extractor, text: str, title: str = "",
                           long_doc_chars: int = 6000, max_sections: int = 6) -> Extraction:
    """One shot for normal docs; section-by-section union for very long ones, so the
    extractor never just truncates a long article. Shared by the ingest pipeline
    (Ingestor._extract_text) and the `extract-dump` tool so they can't drift."""
    if len(text) <= long_doc_chars:
        return extractor.extract_text(text, title)
    merged = Extraction()
    for i in range(0, min(len(text), long_doc_chars * max_sections), long_doc_chars):
        part = extractor.extract_text(text[i:i + long_doc_chars], title if i == 0 else "")
        merged.merge(part)
    return merged


# --------------------------------------------------------------------------- #
# Scripted (deterministic, for the synthetic temporal demo)
# --------------------------------------------------------------------------- #
class ScriptedExtractor:
    """Deterministic extractor backed by a {episode_text: Extraction} table.

    Used by the synthetic evolving-stream demo (kg/synthetic.py) so the temporal ingest
    logic (open / close / supersede) runs end-to-end OFFLINE on clean, known facts — the
    LLM's job (turning prose into typed facts) is the part we stub, leaving the actual
    thing under test (the graph's evolution) running for real. Unknown text → empty."""
    name = "scripted"

    def __init__(self, table: dict[str, Extraction]):
        self._table = {self._norm(k): v for k, v in table.items()}
        self.meter = UsageMeter()   # always empty — keeps testrun's drain() uniform

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join((text or "").split()).strip().lower()

    def _lookup(self, text: str) -> Extraction:
        return self._table.get(self._norm(text), Extraction())

    def extract_text(self, text: str, title: str = "") -> Extraction:
        return self._lookup(text)

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        return self._lookup(label_hint or image_path or "")

    def extract_url(self, url: str) -> Extraction:
        # Keyed by the URL itself so hermetic tests script a page's record deterministically.
        return self._lookup(url)

    def extract_commit(self, message: str, diff: str) -> Extraction:
        # Keyed by the commit MESSAGE so hermetic tests script a commit's record deterministically.
        return self._lookup(message)

    def extract_repo(self, signals: dict) -> Extraction:
        # Keyed by the repo NAME (the salient repo signal) — same policy as the URL/commit keys.
        return self._lookup((signals or {}).get("name", ""))


# --------------------------------------------------------------------------- #
# Cue-gated hybrid extractor (the default production strategy)
# --------------------------------------------------------------------------- #
class CueGatedExtractor:
    """A free local NLP floor on every entry, plus ONE LLM call ONLY on entries carrying a
    termination / relative-date / identity cue (kg/cues.py). The LLM is the merge BASE so its
    temporal relation fields (status=ended, valid_from/to) survive; the local entities/tags
    union in for recall. The exposed `meter` is the LLM's, so the testrun's per-document cost
    drain captures exactly the escalation spend (the local floor is free). With no
    LLM provider available, escalation is disabled and extraction runs local-only."""
    name = "cue_gated"

    def __init__(self, config: Config):
        from .nlp_extractors import build_nlp_extractor   # lazy: it imports THIS module
        self.config = config
        self.local = build_nlp_extractor(
            getattr(config, "local_backend", "gliner_yake_cooccur"), config)
        self.escalate = (bool(getattr(config, "cue_escalate", True))
                         and llm_available())
        self._llm: OpenAIExtractor | None = None
        self._fallback_meter = UsageMeter()
        self.n_seen = 0
        self.n_escalated = 0
        self.cue_counts: dict[str, int] = {}

    def _llm_ext(self) -> OpenAIExtractor:
        if self._llm is None:
            self._llm = OpenAIExtractor(self.config)
        return self._llm

    @property
    def meter(self) -> UsageMeter:
        return self._llm.meter if self._llm is not None else self._fallback_meter

    def extract_text(self, text: str, title: str = "") -> Extraction:
        self.n_seen += 1
        with prof_span("extract.local_nlp"):
            local = self.local.extract_text(text, title)
        if self.escalate and has_cue(text):
            self.n_escalated += 1
            for kind in cue_kinds(text):
                self.cue_counts[kind] = self.cue_counts.get(kind, 0) + 1
            try:
                llm = self._llm_ext().extract_text(text, title)
            except Exception:  # noqa: BLE001 — never sink ingest on one API error; keep the floor
                return local
            return llm.merge(local)             # LLM base → its temporal fields win
        return local

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        return self.local.extract_image(image_path, label_hint)

    def extract_url(self, url: str) -> Extraction:
        """URL ingest on the keyless floor: escalate to the LLM's subject-scoped extractor
        when a provider is live (URL records benefit strongly from it), else fetch the page
        and run the local floor over its body, with a signal-derived description surface."""
        if self.escalate:
            try:
                return self._llm_ext().extract_url(url)
            except Exception:  # noqa: BLE001 — never sink ingest on one API error
                pass
        signals = fetch_page(url)
        ext = self.local.extract_text(signals.get("body") or "", signals.get("title") or "")
        ext.description = _fallback_url_description(signals)
        ext.source_text = signals.get("body") or None
        ext.page_title = signals.get("title") or signals.get("og_title") or None
        return ext

    def extract_commit(self, message: str, diff: str) -> Extraction:
        """Commit ingest on the keyless floor: escalate to the LLM's first-person commit
        extractor when a provider is live (commit records benefit strongly from it), else run
        the local floor over message+diff with the subject line as the description surface."""
        if self.escalate:
            try:
                return self._llm_ext().extract_commit(message, diff)
            except Exception:  # noqa: BLE001 — never sink ingest on one API error
                pass
        ext = self.local.extract_text(f"{message}\n\n{diff}", "")
        ext.description = _concise_commit_description(message)
        ext.source_text = diff or None
        return ext

    def extract_repo(self, signals: dict) -> Extraction:
        """Repo summary on the keyless floor: escalate to the LLM when live, else run the local
        floor over the assembled signals with a README-derived description surface."""
        if self.escalate:
            try:
                return self._llm_ext().extract_repo(signals)
            except Exception:  # noqa: BLE001
                pass
        ext = self.local.extract_text(
            _assemble_repo_block(signals, self.config.extract_max_chars), signals.get("name", ""))
        ext.description = _fallback_repo_description(signals)
        return ext

    def escalation_summary(self) -> dict:
        rate = (self.n_escalated / self.n_seen) if self.n_seen else 0.0
        return {"seen": self.n_seen, "escalated": self.n_escalated,
                "escalation_rate": round(rate, 3), "cue_counts": dict(self.cue_counts)}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_extractor(config: Config) -> Extractor:
    """Default 'auto' = the LLM extractor on EVERY entry when a provider is live (the user's
    signed-in subscription or an API key — extraction should basically always use it), falling
    back to the keyless CueGatedExtractor local floor when none is. 'cue_gated' = that hybrid
    explicitly (local NLP floor + LLM only on cue-bearing entries). 'llm' = the LLM extractor
    strictly — no provider is an error, never a silent downgrade. Any other value selects a
    pure LLM-free / hybrid NLP backend (kg/nlp_extractors.py). The deterministic
    ScriptedExtractor is constructed directly (demo + tests), never here."""
    backend = getattr(config, "extractor_backend", "auto")
    if backend == "haiku":                 # legacy alias from before the multi-provider rename
        backend = "llm"
    if backend == "auto":
        return OpenAIExtractor(config) if llm_available() else CueGatedExtractor(config)
    if backend == "cue_gated":
        return CueGatedExtractor(config)
    if backend != "llm":
        from .nlp_extractors import build_nlp_extractor
        return build_nlp_extractor(backend, config)
    if not llm_available():
        raise RuntimeError(
            "No LLM provider available (set KG_LLM / OPENAI_API_KEY, or sign in to a codex/"
            "claude CLI). The 'llm' extractor backend is live-only. Use the default 'auto' "
            "backend (falls back to a keyless local NLP floor), or construct a "
            "ScriptedExtractor directly for deterministic tests/demos.")
    return OpenAIExtractor(config)
