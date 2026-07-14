"""Flat as-of / evolution fact path.

kg already stores facts bi-temporally on RELATED_TO edges and renders the
currently-valid ones into context. Two gaps the design flags:

  1. Evolution questions ("how has X changed") need the FULL ordered history —
     including the CLOSED (superseded/ended) windows — not just the open fact.
     kg's `facts_for` keeps only `fact_active` edges, so the closed history it
     stored is never served. `history()` returns the closed+open windows in time
     order so the answer can read the trajectory.

  2. The fact-bearing episodes for a state/evolution question may rank low for the
     bi-encoder. `fact_episodes()` returns the episodes that ASSERTED facts about
     the query-linked entities, so the retriever can guarantee they enter the pool.

Both read straight off the per-instance store (small graphs), so there is no extra
index to maintain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Belief, EdgeType
from .store import GraphStore


def _believed(data: dict) -> bool:
    """A RETRACTED fact was never actually true — it is excluded from EVERY view this
    module serves, including the closed-window history (unlike an ended fact, whose
    closed window is genuine trajectory). Mirrors `fact_active`'s belief check without
    its open/as-of window logic, since history() deliberately serves closed windows."""
    return data.get("belief", Belief.ASSERTED.value) == Belief.ASSERTED.value


@dataclass
class FactLine:
    """A relationship line (open or closed) for the answer context: the rendered form
    the prompt/CLI reads plus the structured fields the wire Fact object (PROTOCOL §3)
    carries. `from_edge` builds a fully-populated line; hand-constructed lines (tests,
    legacy sites) leave the structured extras at their defaults."""
    src: str
    rel: str
    dst: str
    valid_at: str = ""
    invalid_at: str = ""
    episode_id: str = ""
    recorded_at: str = ""
    confidence: float | None = None
    provenance: str = ""
    functional: bool = False
    disputed_by: list = field(default_factory=list)

    @classmethod
    def from_edge(cls, store: GraphStore, src_id: str, dst_id: str,
                  data: dict) -> "FactLine":
        """One line from a RELATED_TO edge in stored orientation: endpoint/predicate
        names resolved through the store, structured extras lifted off the edge data."""
        rel = data.get("rel_tag")
        rel_node = store.get_node(rel) if rel else None
        sn, tn = store.get_node(src_id), store.get_node(dst_id)
        return cls(src=sn.name if sn else src_id,
                   rel=rel_node.name if rel_node else "related_to",
                   dst=tn.name if tn else dst_id,
                   valid_at=data.get("valid_at", ""),
                   invalid_at=data.get("invalid_at", ""),
                   episode_id=data.get("episode_id", ""),
                   recorded_at=data.get("created_at", ""),
                   confidence=data.get("confidence"),
                   provenance=data.get("provenance", ""),
                   functional=bool(rel_node.functional) if rel_node else False,
                   disputed_by=list(data.get("disputed_by") or []))

    def to_row(self) -> dict:
        """The wire Fact object (PROTOCOL §3): structured fields plus this line's
        rendered form. Store-empty strings cross as null per the §3 conventions."""
        return {"source": self.src, "predicate": self.rel, "target": self.dst,
                "status": "ended" if self.invalid_at else "asserted",
                "valid_from": self.valid_at or None,
                "valid_to": self.invalid_at or None,
                "recorded_at": self.recorded_at or None,
                "episode_id": self.episode_id or None,
                "confidence": self.confidence,
                "provenance": (self.provenance or "").lower() or None,
                "functional": self.functional,
                "disputed_by": self.disputed_by or [],
                "rendered": self.render()}

    def render(self) -> str:
        win = []
        # `valid_at` on an open fact is usually just the asserting session's created_at,
        # not a confirmed real-world start date — "since" overclaims precision the extractor
        # never had. Only a CLOSED window (invalid_at present) is a real bi-temporal interval,
        # so "since/until" is reserved for that; an open fact gets the honest "mentioned".
        # (Deliberate deviation from PROTOCOL §5.2's Rust line grammar; the explicit
        # "ended" marker keeps a closed line unmistakable next to the row's status.)
        if self.valid_at and self.invalid_at:
            win.append(f"since {self.valid_at[:10]}")
            win.append(f"until {self.invalid_at[:10]}")
            win.append("ended")
        elif self.valid_at:
            win.append(f"mentioned {self.valid_at[:10]}")
        w = f" ({'; '.join(win)})" if win else ""
        prov = f" [{self.episode_id}]" if self.episode_id else ""
        return f"{self.src} --{self.rel}--> {self.dst}{w}{prov}"


class FactIndex:
    def __init__(self, store: GraphStore):
        self.store = store

    def fact_episodes(self, entity_ids: list[str]) -> set[str]:
        """Episodes that asserted any believed fact (open OR closed) about these entities.
        A retracted fact earns its asserting episode no guaranteed pool slot — the claim
        was never true."""
        eps: set[str] = set()
        for eid in entity_ids:
            for direction in ("out", "in"):
                for _nbr, data in self.store.neighbors(
                        eid, etypes={EdgeType.RELATED_TO}, direction=direction):
                    ep = data.get("episode_id")
                    if ep and _believed(data):
                        eps.add(ep)
        return eps

    def history(self, entity_ids: list[str], limit: int = 40) -> list[FactLine]:
        """Full closed+open fact history touching the entities, ordered by valid-time
        (then transaction order), deduped. This is what an 'evolution' answer reads."""
        rows: list[tuple[str, str, FactLine]] = []
        seen: set[tuple] = set()
        for eid in entity_ids:
            for direction in ("out", "in"):
                for nbr, data in self.store.neighbors(
                        eid, etypes={EdgeType.RELATED_TO}, direction=direction):
                    if not _believed(data):
                        continue
                    src_id, dst_id = (eid, nbr) if direction == "out" else (nbr, eid)
                    key = (src_id, data.get("rel_tag"), dst_id,
                           data.get("valid_at", ""), data.get("invalid_at", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append((
                        data.get("valid_at", ""), data.get("created_at", ""),
                        FactLine.from_edge(self.store, src_id, dst_id, data)))
        # order by valid-time start, then transaction time; "" (unknown) sorts first
        rows.sort(key=lambda r: (r[0] or "", r[1] or ""))
        return [fl for _, _, fl in rows[:limit]]
