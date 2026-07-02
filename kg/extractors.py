"""Extractors (docs/ARCHITECTURE.md §6.4).

Pull `{entities[], tags[], relations[]}` directly from raw content (rev 2 — no
summary step) in one structured-output call, with an optional reflexion recall pass.
Relations are directed and carry open-vocabulary `labels[]` (rev 3) that are
consolidated into canonical relationship-tag nodes downstream.

  * OpenAIExtractor  — the real, live path: gpt-4o-mini forced into a typed tool
                       call, vision for images. Needs OPENAI_API_KEY.
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
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from .config import Config
from .cues import cue_kinds, has_cue
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
    canonical relationship-tag nodes downstream (Canonicalizer.resolve_relation).

    Temporal fields (docs/TEMPORAL.md §6): `status` is the polarity — "asserted" (the
    relationship holds) or "ended" (it terminated / no longer holds, from cues like
    "former", "ex-", "no longer"). `valid_from` / `valid_to` are OPTIONAL stated bounds
    (ISO strings); empty means "unknown / as of this episode"."""
    source: str
    target: str
    labels: list[str] = field(default_factory=list)
    provenance: Provenance = Provenance.EXTRACTED
    confidence: float = 0.8
    status: str = "asserted"          # "asserted" | "ended"
    valid_from: str = ""
    valid_to: str = ""


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
                                "'member_of', 'located_in', 'parent_of'. Use the BASE "
                                "predicate even for past relationships (use 'works_with' + "
                                "status 'ended', NOT 'former_colleague'). Consolidated "
                                "automatically.",
                            },
                            "status": {
                                "type": "string", "enum": ["asserted", "ended"],
                                "description": "'asserted' if the relationship holds; 'ended' "
                                "if the text says it terminated (former, ex-, no longer, left, "
                                "until X). Default 'asserted'.",
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
                "description": {"type": "string",
                                "description": "One-line description (images only)."},
            },
            "required": ["entities", "tags"],
        },
    },
}


_TERM_PREFIX = re.compile(r"^(former|formerly|ex|past|no[\s_-]?longer|used[\s_-]?to|once)[\s_-]+",
                          re.I)


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
        base = _TERM_PREFIX.sub("", lab.strip())
        if base != lab.strip():
            ended = True
        out.append(base or lab.strip())
    return list(dict.fromkeys(l for l in out if l)), ended


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
        labels, ended = _normalize_termination(_coerce_labels(r))
        if not s or not t or not labels:
            continue
        status = "ended" if (ended or str(r.get("status", "")).lower() == "ended") else "asserted"
        rels.append(ExtractedRelation(
            source=s, target=t, labels=labels,
            provenance=Provenance.EXTRACTED,
            confidence=float(r.get("confidence", 0.8)),
            status=status, valid_from=str(r.get("valid_from", "") or ""),
            valid_to=str(r.get("valid_to", "") or ""),
        ))
    return Extraction(entities=ents, tags=tags, relations=rels,
                      description=(payload.get("description") or None))


# --------------------------------------------------------------------------- #
# OpenAI (real)
# --------------------------------------------------------------------------- #
class OpenAIExtractor:
    name = "gpt4o_mini"

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
        "   - TIME. If the text says a relationship ENDED (former, ex-, no longer, left, "
        "until X), emit the BASE predicate with status 'ended' — never coin 'former_*' or "
        "'ex_*'. Set valid_from/valid_to ONLY when the text states a date; otherwise leave "
        "them empty (it defaults to 'as of this content'). Do not guess dates.\n"
        "   - Few or no relations is fine if the content doesn't clearly state them.\n\n"
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
        import openai
        self.config = config
        self.client = openai.OpenAI()  # reads OPENAI_API_KEY
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
        msg = self.client.chat.completions.create(
            model=self.config.llm_model,
            max_tokens=self.config.extract_max_tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": content_blocks},
            ],
            tools=[GRAPH_TOOL],
            tool_choice={"type": "function", "function": {"name": "emit_graph"}},
        )
        self.meter.record("extract", self.config.llm_model, msg)
        tc = getattr(msg.choices[0].message, "tool_calls", None) if msg.choices else None
        if tc and tc[0].function.name == "emit_graph":
            return _parse_tool_payload(json.loads(tc[0].function.arguments))
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
        # Per-call input cap (config-driven). Sectioning keeps each slice <= long_doc_chars,
        # and extract_max_chars >= long_doc_chars, so this never truncates a section. It only
        # bites a single un-sectioned call whose text exceeds the cap.
        first = self._call([{"type": "text", "text": header + text[:self.config.extract_max_chars]}])
        return self._reflexion(text, first)

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


# --------------------------------------------------------------------------- #
# Cue-gated hybrid extractor (the default production strategy)
# --------------------------------------------------------------------------- #
class CueGatedExtractor:
    """A free local NLP floor on every entry, plus ONE Haiku call ONLY on entries carrying a
    termination / relative-date / identity cue (kg/cues.py). Haiku is the merge BASE so its
    temporal relation fields (status=ended, valid_from/to) survive; the local entities/tags
    union in for recall. The exposed `meter` is Haiku's, so the testrun's per-document cost
    drain captures exactly the escalation spend (the local floor is free). With no
    OPENAI_API_KEY, escalation is disabled and extraction runs local-only."""
    name = "cue_gated"

    def __init__(self, config: Config):
        from .nlp_extractors import build_nlp_extractor   # lazy: it imports THIS module
        self.config = config
        self.local = build_nlp_extractor(
            getattr(config, "local_backend", "gliner_yake_cooccur"), config)
        self.escalate = (bool(getattr(config, "cue_escalate", True))
                         and bool(os.environ.get("OPENAI_API_KEY")))
        self._haiku: OpenAIExtractor | None = None
        self._fallback_meter = UsageMeter()
        self.n_seen = 0
        self.n_escalated = 0
        self.cue_counts: dict[str, int] = {}

    def _haiku_ext(self) -> OpenAIExtractor:
        if self._haiku is None:
            self._haiku = OpenAIExtractor(self.config)
        return self._haiku

    @property
    def meter(self) -> UsageMeter:
        return self._haiku.meter if self._haiku is not None else self._fallback_meter

    def extract_text(self, text: str, title: str = "") -> Extraction:
        self.n_seen += 1
        local = self.local.extract_text(text, title)
        if self.escalate and has_cue(text):
            self.n_escalated += 1
            for kind in cue_kinds(text):
                self.cue_counts[kind] = self.cue_counts.get(kind, 0) + 1
            try:
                haiku = self._haiku_ext().extract_text(text, title)
            except Exception:  # noqa: BLE001 — never sink ingest on one API error; keep the floor
                return local
            return haiku.merge(local)            # Haiku base → its temporal fields win
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
    """Default 'cue_gated' = a local NLP floor + a Haiku call only on cue-bearing entries
    (escalation needs OPENAI_API_KEY; without it, extraction runs local-only). 'haiku'/
    'auto' = full gpt-4o-mini on every entry (needs the key). Any other value selects a
    pure LLM-free / hybrid NLP backend (kg/nlp_extractors.py). The deterministic
    ScriptedExtractor is constructed directly (demo + tests), never here."""
    backend = getattr(config, "extractor_backend", "cue_gated")
    if backend == "cue_gated":
        return CueGatedExtractor(config)
    if backend not in ("haiku", "auto"):
        from .nlp_extractors import build_nlp_extractor
        return build_nlp_extractor(backend, config)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "No OPENAI_API_KEY found. The 'gpt4o_mini' extractor is live-only. Use the default "
            "'cue_gated' backend (a local NLP floor runs keyless), or construct a "
            "ScriptedExtractor directly for deterministic tests/demos.")
    return OpenAIExtractor(config)
