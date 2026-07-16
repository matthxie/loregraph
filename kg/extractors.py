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
import re
from dataclasses import dataclass, field
from typing import Protocol

from .backoff import call_with_backoff
from .config import Config
from .cues import cue_kinds, has_cue
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
    description: str | None = None  # images: the one-line VLM description
    # How many pure-date entities/tags/relation-endpoints the deterministic post-filter
    # removed (_filter_date_terms) — surfaced on the ingest report so the filter is
    # observable, never silent.
    date_drops: int = 0

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
        self.date_drops += other.date_drops
        return self


class Extractor(Protocol):
    name: str

    # ref_date: the EPISODE's event date (item.created_at, "" when unknown). LLM backends
    # hand it to the model so stated partial/relative dates ("January 2nd", "last March")
    # resolve to ISO valid_from/valid_to; local NLP backends accept and ignore it.
    def extract_text(self, text: str, title: str = "", ref_date: str = "") -> Extraction: ...
    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction: ...


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
                    "description": "Salient nameable entities. Dates, times, months, "
                    "weekdays and years are NEVER entities — temporal information goes "
                    "ONLY in a relation's valid_from/valid_to or a fact's date field.",
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
                         "description": "5-12 lowercase topical tags. Never a date or "
                         "time expression."},
                "relations": {
                    "type": "array",
                    "description": "Directed relationships between entities.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string",
                                       "description": "an entity or tag name — NEVER a "
                                       "date/time expression (dates go in valid_from/"
                                       "valid_to)"},
                            "target": {"type": "string",
                                       "description": "an entity or tag name — NEVER a "
                                       "date/time expression (dates go in valid_from/"
                                       "valid_to)"},
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


# --------------------------------------------------------------------------- #
# Pure-date detection + post-filter (defense in depth: small models emit dates as
# entities / relation endpoints no matter what the prompt says — e.g. the observed
# `St. Mary's Church --attend--> January 2nd`. Deterministic, no LLM.)
# --------------------------------------------------------------------------- #
_MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_PAT = (r"(?:january|february|march|april|may|june|july|august|september|"
              r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)")
_YEAR_PAT = r"(?:1[89]\d{2}|20\d{2})"

# Matches a string that is a PURE date/time expression, whole-string, conservative on
# purpose: bare month names ("May", "August") only count as dates when qualified
# ("last March") or carrying a day/year, so person names are never swallowed; season/
# period words ("spring", "week") require a last/next/this qualifier for the same reason.
_DATE_EXPR_RE = re.compile(
    r"""^(?:the\s+)?(?:(?P<qual>last|next|this|early|late|mid)[\s-]+)?(?:
          (?P<iso>""" + _YEAR_PAT + r"""-[01]\d(?:-[0-3]\d)?)
        | (?P<year>""" + _YEAR_PAT + r""")
        | (?P<month>""" + _MONTH_PAT + r""")\.?
          (?:\s+(?P<day>[0-3]?\d)(?:st|nd|rd|th)?)?
          (?:,?\s+(?P<myear>""" + _YEAR_PAT + r"""))?
        | (?P<day2>[0-3]?\d)(?:st|nd|rd|th)?\s+(?:of\s+)?
          (?P<month2>""" + _MONTH_PAT + r""")\.?
          (?:,?\s+(?P<myear2>""" + _YEAR_PAT + r"""))?
        | (?P<weekday>(?:mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)day)s?
        | (?P<relword>yesterday|today|tonight|tomorrow)
        | (?P<period>week|weekend|month|year|decade|night|morning|afternoon|evening|
                     spring|summer|fall|autumn|winter)
    )$""", re.I | re.X)


def _match_date_expr(value: str) -> "re.Match | None":
    """Whole-string match if `value` is a pure date/time expression, else None."""
    m = _DATE_EXPR_RE.match(" ".join((value or "").split()))
    if not m:
        return None
    qual = m.group("qual")
    # bare month with no qualifier/day/year is likely a name ("May", "August") — keep it
    if m.group("month") and not (qual or m.group("day") or m.group("myear")):
        return None
    if m.group("period") and not qual:      # bare "spring"/"week" — keep it
        return None
    day = m.group("day") or m.group("day2")
    if day and not 1 <= int(day) <= 31:
        return None
    return m


def _is_pure_date(value: str) -> bool:
    return _match_date_expr(value) is not None


def _resolve_date_iso(value: str, ref_date: str) -> str:
    """Resolve a pure-date expression to an ISO date/year string, using `ref_date` (the
    episode's event date) for the missing pieces. Returns "" when it can't be resolved
    without guessing (weekdays, 'yesterday', 'last week', no reference year, …)."""
    m = _match_date_expr(value)
    if not m:
        return ""
    if m.group("iso"):
        return m.group("iso")
    if m.group("year"):
        return m.group("year")
    month = m.group("month") or m.group("month2")
    if not month:
        return ""                        # weekday / relative word — never guess
    mnum = _MONTH_NUM[month.lower().rstrip(".")]
    day = m.group("day") or m.group("day2")
    year = m.group("myear") or m.group("myear2")
    if not year:
        ref = re.match(r"(\d{4})(?:-([01]\d))?", ref_date or "")
        if not ref:
            return ""
        ry, rm = int(ref.group(1)), int(ref.group(2) or 0)
        qual = (m.group("qual") or "").lower()
        if qual == "last" and rm and mnum >= rm:
            year = str(ry - 1)           # "last March" said in Jan–Mar → previous year
        elif qual == "next" and rm and mnum <= rm:
            year = str(ry + 1)
        else:
            year = str(ry)
    return (f"{year}-{mnum:02d}-{int(day):02d}" if day else f"{year}-{mnum:02d}")


def _filter_date_terms(ext: Extraction, ref_date: str = "") -> Extraction:
    """Deterministic post-filter: dates are never graph terms. Drops pure-date entities
    and tags (and anything typed EntityType.DATE), and drops any relation whose endpoint
    is a pure-date expression. Before a dated relation is dropped, its date is SALVAGED:
    if it resolves against `ref_date`, it fills the empty valid_from of a surviving
    relation about the same non-date endpoint (same label when one matches, else the
    single candidate). Everything removed is counted in ext.date_drops."""
    drops = 0
    kept_ents = []
    date_ent_names: set[str] = set()     # dropped date-TYPED surfaces ("the summer of 1969")
    for e in ext.entities:
        if e.type == EntityType.DATE or _is_pure_date(e.name):
            drops += 1
            date_ent_names.add(e.name.strip().lower())
        else:
            kept_ents.append(e)
    kept_tags = []
    for t in ext.tags:
        if _is_pure_date(t):
            drops += 1
        else:
            kept_tags.append(t)
    kept_rels: list[ExtractedRelation] = []
    salvage: list[tuple[str, set[str], str]] = []   # (anchor_lower, labels_lower, iso)
    for r in ext.relations:
        s_date = _is_pure_date(r.source) or r.source.strip().lower() in date_ent_names
        t_date = _is_pure_date(r.target) or r.target.strip().lower() in date_ent_names
        if not s_date and not t_date:
            kept_rels.append(r)
            continue
        drops += 1
        if s_date and t_date:
            continue
        date_surface = r.source if s_date else r.target
        anchor = (r.target if s_date else r.source).strip().lower()
        if not r.valid_from and anchor:
            iso = _resolve_date_iso(date_surface, ref_date)
            if iso:
                salvage.append((anchor, {l.lower() for l in r.labels}, iso))
    for anchor, labels, iso in salvage:
        cands = [r for r in kept_rels if not r.valid_from
                 and anchor in (r.source.strip().lower(), r.target.strip().lower())]
        labelled = [r for r in cands if labels & {l.lower() for l in r.labels}]
        for r in (labelled or (cands if len(cands) == 1 else [])):
            r.valid_from = iso
    ext.entities, ext.tags, ext.relations = kept_ents, kept_tags, kept_rels
    ext.date_drops += drops
    return ext


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
        "Do not invent entities not in the content. A handful is fine; do not pad.\n"
        "   Dates, times, months, weekdays and years are NEVER entities — temporal "
        "information goes ONLY in a relation's valid_from/valid_to or a fact's date "
        "field, never in entities or tags.\n\n"
        "2) TAGS. Emit 5-12 lowercase topical tags describing what the content is ABOUT "
        "(themes, not entities). Never a date or time expression.\n\n"
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
        "content'). Do not guess dates. A date or time expression is NEVER a relation "
        "endpoint: never emit e.g. source \"the church\", target \"January 2nd\", label "
        "\"attended\" — the target must be the real entity, and the date goes in that "
        "relation's valid_from.\n"
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

    # Prepended to the USER content (never _SYS, which stays static for the ingest-cache
    # prompt digest) so the model can resolve stated partial/relative dates against the
    # episode's event date. ref_date is item.created_at; "unknown" when the item has none.
    @staticmethod
    def _reference_date_line(ref_date: str) -> str:
        return (f"This content is dated {ref_date or 'unknown'}. Resolve any partial or "
                "relative dates stated in the text (e.g. 'January 2nd', 'last March') "
                "against this date into ISO form for valid_from/valid_to. If the text "
                "states no date, leave them empty — never guess.\n\n")

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

    def _call(self, content_blocks: list, ref_date: str = "") -> Extraction:
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
            return _filter_date_terms(
                _parse_tool_payload(json.loads(tc[0].function.arguments)), ref_date)
        return Extraction()

    def _reflexion(self, text: str, first: Extraction, ref_date: str = "") -> Extraction:
        if not self.config.reflexion:
            return first
        tags = ", ".join(first.tags) or "(none)"
        ents = ", ".join(e.name for e in first.entities) or "(none)"
        facts_found = ", ".join(
            f"{f.subject} {f.predicate} {f.value}{f.unit}" for f in first.facts) or "(none)"
        prompt = (
            self._reference_date_line(ref_date) +
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
            extra = self._call([{"type": "text", "text": prompt}], ref_date)
            return first.merge(extra)
        except Exception:
            return first

    def extract_text(self, text: str, title: str = "", ref_date: str = "") -> Extraction:
        header = self._reference_date_line(ref_date)
        header += f"Title: {title}\n\n" if title else ""
        # Per-call input cap (config-driven). Sectioning keeps each slice <= long_doc_chars,
        # and extract_max_chars >= long_doc_chars, so this never truncates a section. It only
        # bites a single un-sectioned call whose text exceeds the cap.
        first = self._call([{"type": "text", "text": header + text[:self.config.extract_max_chars]}],
                           ref_date)
        return self._reflexion(text, first, ref_date)

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        hint = f" The image may contain: {label_hint}." if label_hint else ""
        blocks = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
            {"type": "text", "text": "Describe this image in one line and extract its "
             "entities and tags." + hint},
        ]
        ext = self._call(blocks)
        if not ext.description:
            ext.description = (label_hint and f"A photo containing {label_hint}.") or "An image."
        return ext


# --------------------------------------------------------------------------- #
# Shared text-extraction helper (sectioning for long docs, §9 risk 4)
# --------------------------------------------------------------------------- #
def extract_text_sectioned(extractor: Extractor, text: str, title: str = "",
                           long_doc_chars: int = 6000, max_sections: int = 6,
                           ref_date: str = "") -> Extraction:
    """One shot for normal docs; section-by-section union for very long ones, so the
    extractor never just truncates a long article. Shared by the ingest pipeline
    (Ingestor._extract_text) and the `extract-dump` tool so they can't drift.
    `ref_date` (the episode's event date) rides along to EVERY section call so long
    docs resolve stated relative dates too."""
    if len(text) <= long_doc_chars:
        return extractor.extract_text(text, title, ref_date=ref_date)
    merged = Extraction()
    for i in range(0, min(len(text), long_doc_chars * max_sections), long_doc_chars):
        part = extractor.extract_text(text[i:i + long_doc_chars], title if i == 0 else "",
                                      ref_date=ref_date)
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

    def extract_text(self, text: str, title: str = "", ref_date: str = "") -> Extraction:
        return self._lookup(text)      # deterministic table — ref_date is irrelevant

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        return self._lookup(label_hint or image_path or "")


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

    def extract_text(self, text: str, title: str = "", ref_date: str = "") -> Extraction:
        self.n_seen += 1
        with prof_span("extract.local_nlp"):
            local = self.local.extract_text(text, title, ref_date=ref_date)
        if self.escalate and has_cue(text):
            self.n_escalated += 1
            for kind in cue_kinds(text):
                self.cue_counts[kind] = self.cue_counts.get(kind, 0) + 1
            try:
                llm = self._llm_ext().extract_text(text, title, ref_date=ref_date)
            except Exception:  # noqa: BLE001 — never sink ingest on one API error; keep the floor
                return local
            return llm.merge(local)             # LLM base → its temporal fields win
        return local

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        return self.local.extract_image(image_path, label_hint)

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
