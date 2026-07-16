"""Temporal fact-ingest logic (docs/TEMPORAL.md §8).

Each incoming relationship fact maps to exactly ONE action over bi-temporal fact edges:

    open      — newly asserted relationship          → new edge [start, ∞]
    confirm   — same open fact seen again            → bump confidence / fill bounds
    close     — termination signal / explicit end    → set the open edge's invalid_at
    supersede — functional predicate, new value      → close the old value, open the new
    backfill  — start learned AFTER an end-first edge → fill the unknown valid_at
    retract   — correction: the fact was NEVER true  → flip belief, drop from every view
    dispute   — an overturning claim too weak to fire → recorded, edge unchanged

So "coworker → ex-coworker" and "Toronto → Berlin" resolve correctly regardless of
document order. The OPEN-WORLD rule is load-bearing: **absence of mention never closes an
edge** — closure requires positive evidence (the temporal analog of link-biased
under-merge). Identity (the entity node) never changes; only its surrounding edge-set does.
"""
from __future__ import annotations

from .models import Belief, Edge, EdgeType, Provenance
from .store import GraphStore


def _dispute_margin(store: GraphStore) -> float:
    """How far BELOW a stored fact's confidence an overturning claim may sit before it is gated
    (docs/TEMPORAL.md). Read off the store's config; 1.0 disables gating entirely."""
    return float(getattr(store.config, "dispute_confidence_margin", 1.0))


def _gated(margin: float, stored_conf: float, incoming_conf: float) -> bool:
    """True → the incoming claim is TOO WEAK to overturn the stored fact, so the close/supersede/
    retract is BLOCKED and recorded as a dispute instead. Gated only when the incoming confidence
    sits more than `margin` below the stored one — an equal, stronger, or only-slightly-weaker
    claim still overturns normally, so the default behaviour is unchanged for same-trust facts."""
    return incoming_conf < stored_conf - margin


def _record_dispute(data: dict, *, episode_id: str, confidence: float, at: str,
                    action: str) -> None:
    """Append the losing (gated) claim to the stored edge's `disputed_by` list so the disagreement
    can be surfaced rather than silently keeping the stronger fact. Deduped on (episode, action)
    so replaying the same low-trust note doesn't pile up identical entries."""
    entry = {"episode": episode_id, "confidence": round(float(confidence), 3),
             "at": at, "action": action}
    disputed = data.setdefault("disputed_by", [])
    if not any(d.get("episode") == episode_id and d.get("action") == action for d in disputed):
        disputed.append(entry)


def apply_fact(store: GraphStore, *, src: str, dst: str, rel_tag: str, status: str, at: str,
               valid_from: str = "", valid_to: str = "",
               provenance: Provenance = Provenance.EXTRACTED, confidence: float = 0.8,
               episode_id: str = "") -> str:
    """Resolve one (src --rel_tag--> dst) fact into the graph; return the action taken.

    `at` is the asserting episode's event time — the default validity boundary when the
    text states no explicit date. Predicate cardinality (functional / symmetric) is read
    off the RelationNode (stamped at canonicalization time).

    CONFIDENCE-GATED closure: a close / supersede / retract fires only when the asserting
    claim is not far below the fact it would overturn (see `_gated`). A gated claim is
    recorded in the stored edge's `disputed_by` instead of destroying the higher-trust
    fact, and returns the action "dispute"."""
    if not src or not dst or src == dst or not rel_tag:
        return "skip"
    rel = store.get_node(rel_tag)
    functional = bool(rel and rel.functional)
    symmetric = bool(rel and rel.symmetric)
    # REPEATABLE: an event-like predicate (visited/purchased/attended) — neither a single-valued
    # state (functional) nor a relationship state (symmetric). Distinct dates = distinct
    # occurrences, so a differently-dated re-assertion opens a new fact instead of collapsing.
    repeatable = not functional and not symmetric
    margin = _dispute_margin(store)
    # symmetric predicates store ONE orientation, so works_with(A,B) == works_with(B,A)
    if symmetric and src > dst:
        src, dst = dst, src

    # ---- RETRACT (correction: never actually true) --------------------------
    if status == "retracted":
        # transaction-time belief flip — distinct from CLOSE (a valid-time end). Flips any
        # recorded (src,dst,rel) fact to belief='retracted' so it leaves every view,
        # current and as-of-T (fact_active rejects non-asserted belief). The open-world
        # rule still holds: this fires only on explicit corrective evidence, never on
        # silence. A retraction far below the recorded fact's confidence is gated (a weak
        # "actually that's wrong" can't erase a strong fact). find_facts never yields
        # already-retracted edges, so every match here is a live belief.
        matches = list(store.find_facts(src, dst, rel_tag, open_only=False))
        if matches:
            retracted = 0
            for _v, gkey, data in matches:
                if _gated(margin, float(data.get("confidence", 0.0)), confidence):
                    _record_dispute(data, episode_id=episode_id, confidence=confidence,
                                    at=at, action="retract")
                else:
                    data["belief"] = Belief.RETRACTED.value
                    if at:
                        data["retracted_at"] = at
                    if episode_id:
                        data["retracted_by_episode"] = episode_id
                    retracted += 1
                store.touch_edge(src, dst, gkey)
            return "retract" if retracted else "dispute"
        # retract-first: nothing on record to correct → record the retracted belief so the
        # mistaken claim's history exists but is never active.
        edge = _fact_edge(src, dst, rel_tag, valid_from, valid_to, provenance,
                          confidence, episode_id, at)
        edge.belief = Belief.RETRACTED
        edge.retracted_at = at
        edge.retracted_by_episode = episode_id
        store.add_edge(edge)
        return "retract_new"

    # ---- CLOSE (termination) ------------------------------------------------
    if status == "ended":
        end = valid_to or at
        open_facts = list(store.find_facts(src, dst, rel_tag, open_only=True))
        if open_facts:
            closed = 0
            for _v, gkey, data in open_facts:
                if _gated(margin, float(data.get("confidence", 0.0)), confidence):
                    _record_dispute(data, episode_id=episode_id, confidence=confidence,
                                    at=end, action="close")
                else:
                    data["invalid_at"] = end
                    data["closed_at"] = end
                    if episode_id:
                        data["closed_by_episode"] = episode_id
                    closed += 1
                store.touch_edge(src, dst, gkey)
            return "close" if closed else "dispute"
        # end-first: we learned it ended before we ever recorded it holding → a closed
        # edge with unknown start (never fabricate a valid_from).
        store.add_edge(_fact_edge(src, dst, rel_tag, valid_from, end, provenance,
                                  confidence, episode_id, at))
        return "open_closed"

    # ---- asserted -----------------------------------------------------------
    start = valid_from or at

    # SUPERSEDE: a single-valued predicate's new value closes any open value with a
    # DIFFERENT target (you can't live in two cities) before the new one opens. A new value
    # far weaker than the standing one is gated → both stay open (a disputed functional
    # fact), rather than a weak claim evicting the trusted value.
    if functional and symmetric:
        # BOTH functional and symmetric (spouse_of / married_to): the fact is stored in ONE
        # pinned orientation, so the old value can sit as an edge OUT of *or* INTO either
        # endpoint of the new pair. Scan every open fact incident to src OR dst in either
        # direction and close any whose endpoints aren't exactly {src, dst} — otherwise a
        # later marriage to a new person would leave the old one open (out-only scan misses it).
        pair = {src, dst}
        for u, v, gkey, data in _incident_open_facts(store, pair, rel_tag):
            if {u, v} != pair:
                if _gated(margin, float(data.get("confidence", 0.0)), confidence):
                    _record_dispute(data, episode_id=episode_id, confidence=confidence,
                                    at=start, action="supersede")
                else:
                    data["invalid_at"] = start
                    data["closed_at"] = start
                    if episode_id:
                        data["closed_by_episode"] = episode_id
                store.touch_edge(u, v, gkey)
    elif functional:
        for v, gkey, data in list(store.find_facts(src, rel_tag=rel_tag, open_only=True)):
            if v != dst:
                if _gated(margin, float(data.get("confidence", 0.0)), confidence):
                    _record_dispute(data, episode_id=episode_id, confidence=confidence,
                                    at=start, action="supersede")
                else:
                    data["invalid_at"] = start
                    data["closed_at"] = start
                    if episode_id:
                        data["closed_by_episode"] = episode_id
                store.touch_edge(src, v, gkey)

    # EVENT-SHAPED assertion (config.event_facts; docs/PIPELINE.md sharp edge #1): a dated
    # occurrence, not a standing state — stored CLOSED so it can never masquerade as a
    # currently-true fact. Classified with NO LLM: (a) the predicate is in the event
    # lexicon (stamped on the RelationNode at canonicalization), or (b) the assertion
    # arrives with BOTH bounds stated — a "[was in] Japan Nov 1-14" interval is
    # event-shaped by construction whatever its predicate. Runs AFTER supersede so a
    # bounded functional fact still displaces a standing value exactly as today.
    event_shaped = bool(rel and getattr(rel, "event", False)) or \
        bool(valid_from and valid_to)
    if event_shaped and start and getattr(store.config, "event_facts", False):
        # CONFIRM-ON-CLOSED dedup: the generic confirm below only matches OPEN facts, so a
        # same-day re-mention of a closed occurrence would duplicate it. Match ANY believed
        # edge (closed occurrence or legacy open event) with the same valid_at instead; a
        # DIFFERENT date is a genuinely new occurrence and opens a new closed edge.
        same_dated = [(v, gkey, data) for v, gkey, data
                      in store.find_facts(src, dst, rel_tag, open_only=False)
                      if data.get("valid_at", "") == start]
        if same_dated:
            for _v, gkey, data in same_dated:
                data["confidence"] = max(float(data.get("confidence", 0.0)), confidence)
                if episode_id:
                    confirmed = data.setdefault("confirmed_by", [])
                    if episode_id not in confirmed:
                        confirmed.append(episode_id)
                store.touch_edge(src, dst, gkey)
            return "confirm"
        # point event: [d, d]; an explicit bounded [d1, d2] passes through unchanged. The
        # closed edge leaves the current view by design (fact_active untouched) — it is
        # served by the HISTORY/delta block and rendered as an occurrence via the edge's
        # event flag (kg/facts.py FactLine.render).
        store.add_edge(_fact_edge(src, dst, rel_tag, start, valid_to or start,
                                  provenance, confidence, episode_id, at, event=True))
        return "open"

    # CONFIRM: an already-open (src,dst,rel) fact — strengthen, don't duplicate. BUT: for a
    # REPEATABLE predicate, if this occurrence carries an explicit date DIFFERENT from every
    # existing open occurrence's date, it is a genuinely new, separately-dated occurrence
    # (a 2nd visit/purchase/class), not a restatement of the first — EARLIER or LATER — so fall
    # through to OPEN instead of collapsing it (docs: per-occurrence events). For a non-repeatable
    # predicate (functional/symmetric state) a differing date is the opposite signal: the SAME
    # fact began earlier, handled by the confirm branch's widen-to-earliest below.
    open_existing = list(store.find_facts(src, dst, rel_tag, open_only=True))
    new_dated_occurrence = repeatable and bool(valid_from) and open_existing and all(
        data.get("valid_at") and valid_from != data.get("valid_at")
        for _v, _gkey, data in open_existing)
    if open_existing and not new_dated_occurrence:
        for _v, gkey, data in open_existing:
            old_valid = data.get("valid_at", "")
            data["confidence"] = max(float(data.get("confidence", 0.0)), confidence)
            # widen the validity window to the EARLIEST known start: fill an empty valid_at,
            # and if an explicit valid_from arrives that predates the stored start, take the
            # min (ISO lexical order = chronological). This makes a late-arriving earlier
            # start order-independent instead of being ignored once any start was recorded.
            if valid_from:
                cur = data.get("valid_at") or ""
                if not cur or valid_from < cur:
                    data["valid_at"] = valid_from
            if valid_to and not data.get("invalid_at"):
                data["invalid_at"] = valid_to
            # confirmation provenance: every episode that restated the fact (deduped)
            if episode_id:
                confirmed = data.setdefault("confirmed_by", [])
                if episode_id not in confirmed:
                    confirmed.append(episode_id)
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


def _incident_open_facts(store: GraphStore, endpoints: set[str], rel_tag: str):
    """Yield (u, v, gkey, data) for every still-open, still-believed RELATED_TO(rel_tag)
    edge incident to ANY of `endpoints`, in either orientation. Used by the
    functional+symmetric supersede scan, where the old value can hang off either side of a
    pinned edge. Deduped by (u, v, gkey) so a pair both of whose endpoints are in the set
    is visited once. Retracted edges are skipped for the same reason find_facts skips them:
    a never-true fact must not be re-closed or re-disputed."""
    seen = set()
    for node in endpoints:
        if node not in store.g:
            continue
        for adj, out in ((store.g.succ[node], True), (store.g.pred[node], False)):
            for other, edges in adj.items():
                u, v = (node, other) if out else (other, node)
                for gkey, data in edges.items():
                    if data.get("etype") != EdgeType.RELATED_TO.value:
                        continue
                    if data.get("rel_tag") != rel_tag:
                        continue
                    if data.get("invalid_at", ""):
                        continue  # open only
                    if data.get("belief", Belief.ASSERTED.value) == Belief.RETRACTED.value:
                        continue
                    tag = (u, v, gkey)
                    if tag in seen:
                        continue
                    seen.add(tag)
                    yield u, v, gkey, data


def _fact_edge(src, dst, rel_tag, valid_at, invalid_at, provenance, confidence,
               episode_id, at, event: bool = False) -> Edge:
    return Edge(src=src, dst=dst, etype=EdgeType.RELATED_TO, provenance=provenance,
                confidence=confidence, weight=confidence, rel_tag=rel_tag,
                valid_at=valid_at, invalid_at=invalid_at, belief=Belief.ASSERTED,
                episode_id=episode_id, created_at=at, event=event)
