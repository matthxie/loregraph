"""Fact-line embeddings — statement-granularity retrieval vectors (config.fact_vectors).

MOTIVATION (docs/OFFLINE_EVAL.md Round 7a, sharp edge #6). Retrieval scores episode
chunks by topical density, but the gold evidence for many questions is a single
answer-bearing statement made in passing ("what's my blood type?" mentioned once while
booking a flight) — structurally unfindable at chunk granularity, diluted by the
surrounding text. This module makes each FACT a first-class retrieval target: a short,
undiluted surface string per believed RELATED_TO edge, embedded locally (bge-small, $0)
and stored under vector kind="fact". This is the STORAGE layer only; the seeder lane that
consumes these vectors is a follow-up.

Two surface families, both kind="fact", distinguished by the node-id NAMESPACE
(the vectors table has no flag column, so the id prefix carries the distinction):

  STATEMENT  `fact:<h>`     — "<src name> <rel name> <dst name>", one per believed edge,
                              deduped by surface TEXT. Parallel dated occurrences of the
                              same (src,rel,dst) collapse to ONE surface/vector — the
                              design rule "embeddings attach to surfaces that don't
                              mutate" applied to fact lines.
  AGGREGATE  `factagg:<h>`  — a DISTILLED frequency line per (src,rel,dst) group with
                              n_occurrences > 1: "<src> <rel> <dst> N times from <first>
                              to <last>". The disposition/frequency retrieval target
                              ("what do I like to do?" -> `went_to the park 5 times ...`).

KEYING CHOICE — by SURFACE HASH, not by edge. The vectors table is keyed (node_id, kind),
so any stable string id works. Hashing the surface text gives free dedup: five park-visit
edges share one surface, hence one `fact:<h>` id, hence one vector — exactly the prompt's
"dedupe embeddings by surface text". The surface->id map is deterministic and re-derivable
from the graph at query time via `current_surfaces`, so the seeder can map a vector hit
back to its fact(s) without a stored side-table. Keying by edge would instead store one
identical vector per parallel occurrence, which the dedup rule forbids.

RENAME HANDLING. A surface's text changes only when canonicalization renames an endpoint
(a merge). Because the id is the surface hash, a rename yields a NEW id (embedded) and
ORPHANS the old id (pruned) — the "re-embed on flush for surfaces whose text changed"
rule falls out for free. `sync_fact_vectors(prune=True)` (ingest path) reconciles both
directions and counts the churn; `prune=False` (backfill path) is purely additive.
"""
from __future__ import annotations

import hashlib

from .embedders import Embedder
from .facts import FactLine
from .models import Belief, EdgeType
from .store import GraphStore

FACT_KIND = "fact"          # single vector kind for both families
_STMT_PREFIX = "fact:"      # statement-surface node-id namespace
_AGG_PREFIX = "factagg:"    # distilled-aggregate node-id namespace


def _surface_id(prefix: str, surface: str) -> str:
    """Stable node id = namespace prefix + a truncated SHA-256 of the surface text.
    Deterministic and content-addressed: identical surface -> identical id, so parallel
    occurrences and re-derivations dedupe to one vector."""
    return prefix + hashlib.sha256(surface.encode("utf-8")).hexdigest()[:16]


def statement_surface(store: GraphStore, src_id: str, dst_id: str, data: dict) -> str:
    """The deterministic, name-resolved surface for one fact edge: "<src> <rel> <dst>",
    no dates/ids/window grammar. Reuses FactLine's name resolution (the same resolver the
    context/CLI render through) so a fact's surface matches how it's spoken elsewhere."""
    fl = FactLine.from_edge(store, src_id, dst_id, data)
    return f"{fl.src} {fl.rel} {fl.dst}"


def _believed_related(store: GraphStore):
    """Every believed (non-retracted) RELATED_TO edge in stored orientation. Retracted
    facts (never true) earn no retrieval target — mirrors facts._believed / agg_view."""
    for u, v, d in store.g.edges(data=True):
        if d.get("etype") != EdgeType.RELATED_TO.value:
            continue
        if d.get("belief", Belief.ASSERTED.value) != Belief.ASSERTED.value:
            continue
        yield u, v, d


def current_surfaces(store: GraphStore) -> tuple[dict[str, str], dict[str, str]]:
    """({stmt_id: surface}, {agg_id: surface}) for the store's current believed fact set.

    Statements dedupe by surface text. Aggregates reuse Round 6b's tally grouping
    definition (kg/rag.py `_graph_tallies`): parallel edges grouped by (src, rel_tag, dst);
    n_occurrences = SUM(1 + len(confirmed_by)); believed-only; first/last = earliest/latest
    non-empty valid_at — and an aggregate is emitted only for a group with n_occurrences>1.
    """
    stmt: dict[str, str] = {}
    groups: dict[tuple, dict] = {}
    for u, v, d in _believed_related(store):
        surface = statement_surface(store, u, v, d)
        stmt[_surface_id(_STMT_PREFIX, surface)] = surface
        key = (u, d.get("rel_tag"), v)
        g = groups.setdefault(key, {"n": 0, "dates": []})
        g["n"] += 1 + len(d.get("confirmed_by") or [])       # each parallel edge + its confirms
        val = d.get("valid_at", "")
        if val:
            g["dates"].append(val[:10])

    agg: dict[str, str] = {}
    for (u, rel, v), g in groups.items():
        if g["n"] <= 1:                                       # distilled lines are for RECURRENCE
            continue
        sn, dn = store.get_node(u), store.get_node(v)
        rn = store.get_node(rel) if rel else None
        src = sn.name if sn else u
        dst = dn.name if dn else v
        rname = rn.name if rn else "related_to"
        n = g["n"]
        if g["dates"]:
            lo, hi = min(g["dates"]), max(g["dates"])
            surface = f"{src} {rname} {dst} {n} times from {lo} to {hi}"
        else:                                                 # undated confirm-collapse: no span
            surface = f"{src} {rname} {dst} {n} times"
        agg[_surface_id(_AGG_PREFIX, surface)] = surface
    return stmt, agg


def _edge_episodes(data: dict) -> set[str]:
    """Provenance episodes for one fact edge: the asserting episode plus every episode that
    confirmed it (same-key re-mentions collapse onto `confirmed_by`). These are the CHUNK ids
    the statement was extracted from — the needle chunks the fact lane pulls into the pool."""
    eps: set[str] = set()
    ep = data.get("episode_id")
    if ep:
        eps.add(ep)
    eps.update(data.get("confirmed_by") or [])
    return eps


def fact_provenance(store: GraphStore, hit_ids) -> dict[str, dict]:
    """Map fact/aggregate vector ids (hits from a kind="fact" search) back to the graph.

    Returns {hit_id: {"surface", "stmt_surface", "episodes": set, "entities": set}}. The
    surface→id map is deterministic and re-derivable (kg/fact_vectors keying choice), so no
    side-table is stored: one walk of the believed edges recomputes each surface's id and, for
    any id in `hit_ids`, unions the provenance of the fact(s) behind it. Because parallel dated
    occurrences dedupe to one statement surface, a single statement hit yields ALL its
    occurrences' provenance; an aggregate hit yields its whole (src,rel,dst) group's. Mirrors
    `current_surfaces` exactly so a hit always resolves. `stmt_surface` is the plain
    "<src> <rel> <dst>" line (== the statement surface / an aggregate's group statement),
    the key the FACTS section marks [matched] against."""
    hit_ids = set(hit_ids)
    if not hit_ids:
        return {}
    out: dict[str, dict] = {}
    groups: dict[tuple, dict] = {}
    for u, v, d in _believed_related(store):
        surface = statement_surface(store, u, v, d)
        sid = _surface_id(_STMT_PREFIX, surface)
        eps = _edge_episodes(d)
        if sid in hit_ids:
            rec = out.setdefault(sid, {"surface": surface, "stmt_surface": surface,
                                       "episodes": set(), "entities": set()})
            rec["episodes"].update(eps)
            rec["entities"].update((u, v))
        key = (u, d.get("rel_tag"), v)
        g = groups.setdefault(key, {"n": 0, "dates": [], "episodes": set(),
                                    "stmt_surface": surface})
        g["n"] += 1 + len(d.get("confirmed_by") or [])
        g["episodes"].update(eps)
        val = d.get("valid_at", "")
        if val:
            g["dates"].append(val[:10])

    for (u, rel, v), g in groups.items():
        if g["n"] <= 1:
            continue
        sn, dn = store.get_node(u), store.get_node(v)
        rn = store.get_node(rel) if rel else None
        src = sn.name if sn else u
        dst = dn.name if dn else v
        rname = rn.name if rn else "related_to"
        n = g["n"]
        if g["dates"]:
            lo, hi = min(g["dates"]), max(g["dates"])
            surface = f"{src} {rname} {dst} {n} times from {lo} to {hi}"
        else:
            surface = f"{src} {rname} {dst} {n} times"
        aid = _surface_id(_AGG_PREFIX, surface)
        if aid in hit_ids:
            rec = out.setdefault(aid, {"surface": surface, "stmt_surface": g["stmt_surface"],
                                       "episodes": set(), "entities": set()})
            rec["episodes"].update(g["episodes"])
            rec["entities"].update((u, v))
    return out


def sync_fact_vectors(store: GraphStore, embedder: Embedder, *, prune: bool = True) -> dict:
    """Reconcile the kind="fact" vector index to the store's current surfaces.

    Embeds only the MISSING surfaces (incremental — embedding cost is proportional to
    new/changed facts, never the whole store). With `prune=True` (ingest path) it also
    drops vectors whose surface no longer exists — the honest cost of surface-hash keying:
    a canonical rename re-embeds the new surface AND orphans the old one. With
    `prune=False` (backfill path) it is purely additive and idempotent.

    Returns counts: added (surfaces embedded this pass), removed (orphans pruned),
    statements / aggregates (current surface totals), total.
    """
    stmt, agg = current_surfaces(store)
    want = {**stmt, **agg}                                    # node_id -> surface
    existing = set(store.vectors.ids(FACT_KIND))
    missing = [nid for nid in want if nid not in existing]
    orphaned = [nid for nid in existing if nid not in want] if prune else []

    if missing:
        surfaces = [want[nid] for nid in missing]
        vecs = embedder.embed(surfaces)
        for nid, vec in zip(missing, vecs):
            store.vectors.add(FACT_KIND, nid, vec)           # marks (kind,nid) dirty via on_add
    for nid in orphaned:
        store.remove_vector(FACT_KIND, nid)

    return {"added": len(missing), "removed": len(orphaned),
            "statements": len(stmt), "aggregates": len(agg), "total": len(want)}


def backfill_fact_vectors(store: GraphStore, embedder: Embedder) -> dict:
    """Load-time / CLI backfill: compute any MISSING fact vectors for an existing store —
    pure local embedding ($0), additive and idempotent, touches only the vectors table
    (never nodes/edges/config), so it cannot invalidate the ingest cache. Lets the cached
    benchmark stores gain fact vectors without a paid re-ingest."""
    return sync_fact_vectors(store, embedder, prune=False)
