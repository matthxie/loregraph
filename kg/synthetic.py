"""A tiny synthetic *evolving* corpus — the Becky / Alex timeline (docs/TEMPORAL.md §10).

The frozen Wikipedia snapshot can't exercise temporal evolution, so this hand-built stream
does: four dated episodes in which Becky's state changes over time. Paired with the
`ScriptedExtractor`, the prose→facts step is deterministic, so the thing actually under
test — the graph's bi-temporal evolution (open / close / supersede) and the as-of-T
retrieval over it — runs fully offline with no LLM.

    from kg.synthetic import becky_stream
    items, table = becky_stream()
    g.extractor = ScriptedExtractor(table)
    g.ingest(items)

Expected outcome after ingest:
    current view  → Becky lives_in Berlin, works_with Dana, employed_by Globex
                    (Toronto superseded 2023, Acme superseded 2024, Alex closed 2024)
    as of 2022    → Becky lives_in Toronto, works_with Alex, employed_by Acme
"""
from __future__ import annotations

from .corpus import CorpusItem
from .extractors import ExtractedEntity, ExtractedRelation, Extraction
from .models import EntityType, Provenance


def _E(name: str, etype: EntityType) -> ExtractedEntity:
    return ExtractedEntity(name=name, type=etype)


def _R(src: str, tgt: str, labels: list[str], status: str = "asserted",
       conf: float = 0.95) -> ExtractedRelation:
    return ExtractedRelation(source=src, target=tgt, labels=labels,
                             provenance=Provenance.EXTRACTED, confidence=conf, status=status)


_P, _PL, _O = EntityType.PERSON, EntityType.PLACE, EntityType.ORG

# (id, created_at, title, text, Extraction)
_EPISODES = [
    ("becky01", "2021-03-15T00:00:00+00:00", "Becky in 2021",
     "Becky lived in Toronto and worked with Alex at Acme Corp.",
     Extraction(
         entities=[_E("Becky", _P), _E("Toronto", _PL), _E("Alex", _P), _E("Acme Corp", _O)],
         tags=["biography", "career", "people"],
         relations=[_R("Becky", "Toronto", ["lives_in"]),
                    _R("Becky", "Alex", ["works_with"]),
                    _R("Becky", "Acme Corp", ["employed_by"])])),

    ("becky02", "2023-06-01T00:00:00+00:00", "Becky relocates",
     "Becky moved to Berlin in 2023 for a new opportunity.",
     Extraction(
         entities=[_E("Becky", _P), _E("Berlin", _PL)],
         tags=["biography", "relocation", "people"],
         relations=[_R("Becky", "Berlin", ["lives_in"])])),

    ("becky03", "2024-01-10T00:00:00+00:00", "Becky and Alex",
     "Becky and Alex are former colleagues; they no longer work together.",
     Extraction(
         entities=[_E("Becky", _P), _E("Alex", _P)],
         tags=["biography", "career", "people"],
         relations=[_R("Becky", "Alex", ["works_with"], status="ended")])),

    ("becky04", "2024-09-20T00:00:00+00:00", "Becky's new role",
     "Becky now works with Dana at Globex.",
     Extraction(
         entities=[_E("Becky", _P), _E("Dana", _P), _E("Globex", _O)],
         tags=["biography", "career", "people"],
         relations=[_R("Becky", "Dana", ["works_with"]),
                    _R("Becky", "Globex", ["employed_by"])])),
]


def becky_stream() -> tuple[list[CorpusItem], dict[str, Extraction]]:
    """Return (items, extraction_table) for the evolving Becky / Alex stream."""
    items, table = [], {}
    for eid, ts, title, text, ext in _EPISODES:
        items.append(CorpusItem(id=eid, modality="text", source_ref=f"synthetic/{eid}",
                                title=title, text=text, created_at=ts))
        table[text] = ext
    return items, table


# A first-person evolving stream (personal-web mode). Each episode's ScriptedExtractor
# entry names the narrator exactly 'me' (lowercase) so resolve_entity('me') routes to the
# single SELF_ENTITY_ID anchor when config.self_entity is on. created_at increases so the
# temporal layer orders the facts deterministically.
_PERSONAL_EPISODES = [
    ("me01", "2025-01-01T00:00:00+00:00", "Coffee with Becky",
     "i had coffee with Becky today.",
     Extraction(
         entities=[_E("me", _P), _E("Becky", _P)],
         tags=["personal", "coffee", "people"],
         relations=[_R("me", "Becky", ["had_coffee_with"])])),

    ("me02", "2025-01-15T00:00:00+00:00", "Meeting Dana",
     "Becky introduced me to Dana.",
     Extraction(
         entities=[_E("Becky", _P), _E("me", _P), _E("Dana", _P)],
         tags=["personal", "people"],
         relations=[_R("Becky", "Dana", ["introduced"]),
                    _R("me", "Dana", ["met"])])),

    ("me03", "2025-02-01T00:00:00+00:00", "Working with Dana",
     "i started working with Dana.",
     Extraction(
         entities=[_E("me", _P), _E("Dana", _P)],
         tags=["personal", "career", "people"],
         relations=[_R("me", "Dana", ["works_with"])])),
]


def personal_stream() -> tuple[list[CorpusItem], dict[str, Extraction]]:
    """Return (items, extraction_table) for the first-person personal-web stream.

    Mirrors becky_stream(): paired with a ScriptedExtractor + config.self_entity=True the
    'me' narrator entity canonicalizes onto SELF_ENTITY_ID, so the self anchor's facts
    (self --had_coffee_with--> Becky, self --works_with--> Dana, …) form deterministically
    OFFLINE."""
    items, table = [], {}
    for eid, ts, title, text, ext in _PERSONAL_EPISODES:
        items.append(CorpusItem(id=eid, modality="text", source_ref=f"synthetic/{eid}",
                                title=title, text=text, created_at=ts))
        table[text] = ext
    return items, table
