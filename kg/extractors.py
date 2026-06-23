"""Extractors (docs/ARCHITECTURE.md §6.4).

Pull `{entities[], tags[], relations[]}` directly from raw content (rev 2 — no
summary step) in one structured-output call, with an optional reflexion recall pass.
Relations are directed and carry open-vocabulary `labels[]` (rev 3) that are
consolidated into canonical relationship-tag nodes downstream.

  * HaikuExtractor   — the real path: Claude Haiku 4.5 forced into a typed tool call,
                       vision for images. Needs ANTHROPIC_API_KEY.
  * HeuristicExtractor — offline deterministic fallback (proper-noun + keyword
                       extraction; images use the COCO manifest label as the VLM
                       stand-in). Lets the whole pipeline run with no API key.

Both return the same `Extraction` object so the ingestion pipeline is backend-blind.
"""
from __future__ import annotations

import base64
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from .config import Config
from .metering import UsageMeter
from .models import EntityType, Provenance

# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class ExtractedEntity:
    name: str
    type: EntityType = EntityType.OTHER


@dataclass
class ExtractedRelation:
    """A directed connection source→target carrying open-vocabulary relationship
    labels (e.g. ["is_friend_of", "works_with"]). The labels are consolidated into
    canonical relationship-tag nodes downstream (Canonicalizer.resolve_relation)."""
    source: str
    target: str
    labels: list[str] = field(default_factory=list)
    provenance: Provenance = Provenance.EXTRACTED
    confidence: float = 0.8


@dataclass
class Extraction:
    entities: list[ExtractedEntity] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    description: str | None = None  # images: the one-line VLM description

    def merge(self, other: "Extraction") -> "Extraction":
        names = {e.name.lower() for e in self.entities}
        for e in other.entities:
            if e.name.lower() not in names:
                self.entities.append(e); names.add(e.name.lower())
        tagset = {t.lower() for t in self.tags}
        for t in other.tags:
            if t.lower() not in tagset:
                self.tags.append(t); tagset.add(t.lower())
        # merge by directed (source, target): union the label sets of duplicates
        by_pair: dict[tuple[str, str], ExtractedRelation] = {}
        for r in self.relations:
            by_pair[(r.source.lower(), r.target.lower())] = r
        for r in other.relations:
            kk = (r.source.lower(), r.target.lower())
            if kk in by_pair:
                have = by_pair[kk]
                for lab in r.labels:
                    if lab not in have.labels:
                        have.labels.append(lab)
            else:
                by_pair[kk] = r
                self.relations.append(r)
        return self


class Extractor(Protocol):
    name: str

    def extract_text(self, text: str, title: str = "") -> Extraction: ...
    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction: ...


# --------------------------------------------------------------------------- #
# Structured-output tool schema (shared by the Haiku path)
# --------------------------------------------------------------------------- #
_ENTITY_ENUM = [t.value for t in EntityType]

GRAPH_TOOL = {
    "name": "emit_graph",
    "description": "Emit the knowledge-graph elements extracted from the content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": _ENTITY_ENUM},
                    },
                    "required": ["name", "type"],
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
                            "'member_of', 'located_in', 'parent_of'. Use natural "
                            "predicate names; they are consolidated automatically.",
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["source", "target", "labels"],
                },
            },
            "description": {"type": "string",
                            "description": "One-line description (images only)."},
        },
        "required": ["entities", "tags"],
    },
}


def _coerce_labels(r: dict) -> list[str]:
    """Open-vocab labels[], with back-compat for an old single `relation` string."""
    labels = [str(x).strip() for x in (r.get("labels") or []) if str(x).strip()]
    if not labels and r.get("relation"):
        labels = [str(r["relation"]).strip()]
    # de-dupe preserving order
    return list(dict.fromkeys(labels))


def _parse_tool_payload(payload: dict) -> Extraction:
    ents = []
    for e in payload.get("entities", []) or []:
        name = (e.get("name") or "").strip()
        if not name:
            continue
        try:
            etype = EntityType(e.get("type", "other"))
        except ValueError:
            etype = EntityType.OTHER
        ents.append(ExtractedEntity(name=name, type=etype))
    tags = [t.strip() for t in (payload.get("tags") or []) if t and t.strip()]
    rels = []
    for r in payload.get("relations", []) or []:
        s, t = (r.get("source") or "").strip(), (r.get("target") or "").strip()
        labels = _coerce_labels(r)
        if not s or not t or not labels:
            continue
        rels.append(ExtractedRelation(
            source=s, target=t, labels=labels,
            provenance=Provenance.EXTRACTED,
            confidence=float(r.get("confidence", 0.8)),
        ))
    return Extraction(entities=ents, tags=tags, relations=rels,
                      description=(payload.get("description") or None))


# --------------------------------------------------------------------------- #
# Haiku (real)
# --------------------------------------------------------------------------- #
class HaikuExtractor:
    name = "haiku"

    # The system prompt + GRAPH_TOOL are kept STATIC and BLIND (no live graph
    # vocabulary is ever injected) — extraction never diffs against existing state;
    # that is a separate downstream concern (canonicalize.py + the L3 tie-breaker).
    # The predicate list below is a FROZEN soft hint, not the growing vocabulary.
    # (No cache_control is set: the prefix is ~1K tokens, under Haiku 4.5's 4096-token
    # minimum cacheable prefix, so it would silently no-op today. Keeping it static
    # makes it cache-eligible if it ever grows past that minimum — e.g. with few-shots.)
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
        "   Prefer the fullest proper name the content uses (\"John F. Kennedy\", not \"JFK\"). "
        "Do not invent entities not in the content. A handful is fine; do not pad.\n\n"
        "2) TAGS. Emit 5-12 lowercase topical tags describing what the content is ABOUT "
        "(themes, not entities).\n\n"
        "3) RELATIONS. Emit the key DIRECTED relationships. RULES:\n"
        "   - Both source and target MUST be entities from step 1, using the EXACT SAME "
        "surface string you wrote there. Never relate something you did not name.\n"
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
        "   - Few or no relations is fine if the content doesn't clearly state them.\n\n"
        "Call emit_graph exactly once."
    )

    def __init__(self, config: Config):
        import anthropic
        self.config = config
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        # per-extractor token/cost accumulator (the test-run dashboard drains it per
        # document); always present so callers stay backend-blind.
        self.meter = UsageMeter()

    def _call(self, content_blocks: list) -> Extraction:
        msg = self.client.messages.create(
            model=self.config.llm_model,
            max_tokens=1500,
            temperature=0,   # reproducibility: the API default is 1.0. temperature is a
                             # valid param on Haiku 4.5 / Sonnet 4.6 (only removed on
                             # Opus 4.7+/Fable 5). The canonicalized topology is what's
                             # reproducible; the raw LLM output is still not bit-exact.
            system=self._SYS,
            tools=[GRAPH_TOOL],
            tool_choice={"type": "tool", "name": "emit_graph"},
            messages=[{"role": "user", "content": content_blocks}],
        )
        # record token usage + cost (no-op for fakes/offline that carry no .usage)
        self.meter.record("extract", self.config.llm_model, msg)
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "emit_graph":
                return _parse_tool_payload(block.input)
        return Extraction()

    def _reflexion(self, text: str, first: Extraction) -> Extraction:
        if not self.config.reflexion:
            return first
        tags = ", ".join(first.tags) or "(none)"
        ents = ", ".join(e.name for e in first.entities) or "(none)"
        prompt = (
            f"Content:\n{text[:4000]}\n\n"
            f"First pass found these tags: {tags}\n"
            f"and these entities: {ents}.\n\n"
            "Now do a focused recall check. List ONLY items you OMITTED:\n"
            "- any salient entity in the content missing above (with its type),\n"
            "- any important topical tag not already emitted,\n"
            "- any clearly-stated directed relationship between entities you missed "
            "(source and target must be named entities; labels read source→target).\n\n"
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
        first = self._call([{"type": "text", "text": header + text[:12000]}])
        return self._reflexion(text, first)

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        hint = f" The image may contain: {label_hint}." if label_hint else ""
        blocks = [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg", "data": data}},
            {"type": "text", "text": "Describe this image in one line and extract its "
             "entities and tags." + hint},
        ]
        ext = self._call(blocks)
        if not ext.description:
            ext.description = (label_hint and f"A photo containing {label_hint}.") or "An image."
        return ext


# --------------------------------------------------------------------------- #
# Heuristic (offline)
# --------------------------------------------------------------------------- #
_STOP = set("""
the a an and or but of to in on at by for with from into over under again further then once
is are was were be been being have has had do does did doing this that these those it its as
i you he she they we me him her them my your his their our who whom which what when where why
how all any both each few more most other some such no nor not only own same so than too very
can will just don should now also which their about after before during between through up down
out off above below new one two first time year years used using use known including part many
""".split())

_PROPER = re.compile(r"\b([A-Z][a-zA-Z0-9'’.-]+(?:\s+[A-Z][a-zA-Z0-9'’.-]+){0,3})\b")
_WORD = re.compile(r"[a-z][a-z0-9-]{2,}")
_ORG_CUES = ("Inc", "Corp", "Ltd", "LLC", "Company", "Association", "University",
             "Institute", "Committee", "Council", "Commission", "League", "Club",
             "Department", "Society", "Foundation", "Group", "Party", "Bank")


def _classify(name: str) -> EntityType:
    if any(cue in name for cue in _ORG_CUES):
        return EntityType.ORG
    toks = name.split()
    if len(toks) == 2 and all(t[0].isupper() for t in toks):
        return EntityType.PERSON
    return EntityType.OTHER


class HeuristicExtractor:
    name = "heuristic"

    def __init__(self, config: Config):
        self.config = config
        # offline: no API call ever happens, so the meter stays empty → $0 / 0 tokens.
        # Present so the ingest pipeline can read `extractor.meter` without branching.
        self.meter = UsageMeter()

    def extract_text(self, text: str, title: str = "") -> Extraction:
        body = f"{title}. {text}" if title else text
        # entities: frequent proper-noun phrases
        cand = Counter()
        for m in _PROPER.finditer(body):
            phrase = m.group(1).strip(" .")
            head = phrase.split()[0]
            if head.lower() in _STOP or len(phrase) < 3:
                continue
            cand[phrase] += 1
        entities, seen = [], set()
        for name, _ in cand.most_common(20):
            low = name.lower()
            if low in seen:
                continue
            seen.add(low)
            entities.append(ExtractedEntity(name=name, type=_classify(name)))
        # tags: frequent topical content words
        words = Counter(w for w in _WORD.findall(body.lower()) if w not in _STOP)
        tags = [w for w, _ in words.most_common(10)]
        # relations: co-occurrence of the top entity with the rest (low-confidence).
        # The offline heuristic can't name a real predicate, so it emits the generic
        # "related_to" label — which still flows through relation consolidation.
        rels = []
        if len(entities) >= 2:
            hub = entities[0].name
            for e in entities[1:6]:
                rels.append(ExtractedRelation(
                    source=hub, target=e.name, labels=["related_to"],
                    provenance=Provenance.INFERRED, confidence=0.4))
        return Extraction(entities=entities, tags=tags, relations=rels)

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        labels = [l.strip() for l in (label_hint or "").split(",") if l.strip()]
        if not labels:
            labels = ["photo"]
        entities = [ExtractedEntity(name=l, type=EntityType.CONCEPT) for l in labels]
        desc = f"A photo containing {', '.join(labels)}."
        return Extraction(entities=entities, tags=labels, description=desc)


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
# Factory
# --------------------------------------------------------------------------- #
def get_extractor(config: Config) -> Extractor:
    choice = config.extractor
    if choice == "haiku":
        return HaikuExtractor(config)
    if choice == "heuristic":
        return HeuristicExtractor(config)
    # auto
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return HaikuExtractor(config)
        except Exception:
            pass
    return HeuristicExtractor(config)
