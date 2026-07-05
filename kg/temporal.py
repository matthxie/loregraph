"""Temporal fact-ingest logic (docs/TEMPORAL.md §8).

Each incoming relationship fact maps to exactly ONE action over bi-temporal fact edges:

    open      — newly asserted relationship          → new edge [start, ∞]
    confirm   — same open fact seen again            → bump confidence / fill bounds
    close     — termination signal / explicit end    → set the open edge's invalid_at
    supersede — functional predicate, new value      → close the old value, open the new
    backfill  — start learned AFTER an end-first edge → fill the unknown valid_at

So "coworker → ex-coworker" and "Toronto → Berlin" resolve correctly regardless of
document order. The OPEN-WORLD rule is load-bearing: **absence of mention never closes an
edge** — closure requires positive evidence (the temporal analog of link-biased
under-merge). Identity (the entity node) never changes; only its surrounding edge-set does.
"""
from __future__ import annotations

from .models import Belief, Edge, EdgeType, Provenance
from .store import GraphStore


def apply_fact(store: GraphStore, *, src: str, dst: str, rel_tag: str, status: str, at: str,
               valid_from: str = "", valid_to: str = "",
               provenance: Provenance = Provenance.EXTRACTED, confidence: float = 0.8,
               episode_id: str = "") -> str:
    """Resolve one (src --rel_tag--> dst) fact into the graph; return the action taken.

    `at` is the asserting episode's event time — the default validity boundary when the
    text states no explicit date. Predicate cardinality (functional / symmetric) is read
    off the RelationNode (stamped at canonicalization time)."""
    if not src or not dst or src == dst or not rel_tag:
        return "skip"
    rel = store.get_node(rel_tag)
    functional = bool(rel and rel.functional)
    symmetric = bool(rel and rel.symmetric)
    # symmetric predicates store ONE orientation, so works_with(A,B) == works_with(B,A)
    if symmetric and src > dst:
        src, dst = dst, src

    # ---- CLOSE (termination) ------------------------------------------------
    if status == "ended":
        end = valid_to or at
        if store.close_facts(src, dst, rel_tag, end):
            return "close"
        # end-first: we learned it ended before we ever recorded it holding → a closed
        # edge with unknown start (never fabricate a valid_from).
        store.add_edge(_fact_edge(src, dst, rel_tag, valid_from, end, provenance,
                                  confidence, episode_id, at))
        return "open_closed"

    # ---- asserted -----------------------------------------------------------
    start = valid_from or at

    # SUPERSEDE: a single-valued predicate's new value closes any open value with a
    # DIFFERENT target (you can't live in two cities) before the new one opens.
    if functional:
        for v, gkey, data in list(store.find_facts(src, rel_tag=rel_tag, open_only=True)):
            if v != dst:
                data["invalid_at"] = start
                store.touch_edge(src, v, gkey)

    # CONFIRM: an already-open (src,dst,rel) fact — strengthen, don't duplicate. BUT: if
    # this occurrence carries an explicit date that matches NONE of the existing open
    # occurrences' dates, it is a genuinely new, separately-dated occurrence of a
    # repeatable predicate (a 2nd visit/purchase/class), not a restatement of the first —
    # fall through to OPEN instead of collapsing it (docs: per-occurrence events).
    open_existing = list(store.find_facts(src, dst, rel_tag, open_only=True))
    new_dated_occurrence = bool(valid_from) and open_existing and all(
        data.get("valid_at") and data.get("valid_at") != valid_from
        for _v, _gkey, data in open_existing)
    if open_existing and not new_dated_occurrence:
        for _v, gkey, data in open_existing:
            old_valid = data.get("valid_at", "")
            data["confidence"] = max(float(data.get("confidence", 0.0)), confidence)
            if not data.get("valid_at") and valid_from:
                data["valid_at"] = valid_from
            if valid_to and not data.get("invalid_at"):
                data["invalid_at"] = valid_to
            store.touch_edge(src, dst, gkey, old_valid_at=old_valid)
        return "confirm"

    # BACKFILL: order-independence — an end-first closed edge with unknown start gets its
    # start filled instead of spawning a duplicate open edge.
    for _v, gkey, data in store.find_facts(src, dst, rel_tag, open_only=False):
        if data.get("invalid_at") and not data.get("valid_at"):
            old_valid = data.get("valid_at", "")
            data["valid_at"] = start
            store.touch_edge(src, dst, gkey, old_valid_at=old_valid)
            return "backfill"

    # OPEN: a brand-new fact.
    store.add_edge(_fact_edge(src, dst, rel_tag, start, valid_to, provenance,
                              confidence, episode_id, at))
    return "open"


def _fact_edge(src, dst, rel_tag, valid_at, invalid_at, provenance, confidence,
               episode_id, at) -> Edge:
    return Edge(src=src, dst=dst, etype=EdgeType.RELATED_TO, provenance=provenance,
                confidence=confidence, weight=confidence, rel_tag=rel_tag,
                valid_at=valid_at, invalid_at=invalid_at, belief=Belief.ASSERTED,
                episode_id=episode_id, created_at=at)
