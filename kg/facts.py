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

from dataclasses import dataclass

from .models import EdgeType
from .store import GraphStore


@dataclass
class FactLine:
    """A rendered relationship line (open or closed) for the answer context."""
    src: str
    rel: str
    dst: str
    valid_at: str = ""
    invalid_at: str = ""
    episode_id: str = ""

    def render(self) -> str:
        win = []
        # `valid_at` on an open fact is usually just the asserting session's created_at,
        # not a confirmed real-world start date — "since" overclaims precision the extractor
        # never had. Only a CLOSED window (invalid_at present) is a real bi-temporal interval,
        # so "since/until" is reserved for that; an open fact gets the honest "mentioned".
        if self.valid_at and self.invalid_at:
            win.append(f"since {self.valid_at[:10]}")
            win.append(f"until {self.invalid_at[:10]}")
        elif self.valid_at:
            win.append(f"mentioned {self.valid_at[:10]}")
        w = f" ({'; '.join(win)})" if win else ""
        prov = f" [{self.episode_id}]" if self.episode_id else ""
        return f"{self.src} --{self.rel}--> {self.dst}{w}{prov}"


class FactIndex:
    def __init__(self, store: GraphStore):
        self.store = store

    def fact_episodes(self, entity_ids: list[str]) -> set[str]:
        """Episodes that asserted any fact (open OR closed) about these entities."""
        eps: set[str] = set()
        for eid in entity_ids:
            for direction in ("out", "in"):
                for _nbr, data in self.store.neighbors(
                        eid, etypes={EdgeType.RELATED_TO}, direction=direction):
                    ep = data.get("episode_id")
                    if ep:
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
                    src_id, dst_id = (eid, nbr) if direction == "out" else (nbr, eid)
                    key = (src_id, data.get("rel_tag"), dst_id,
                           data.get("valid_at", ""), data.get("invalid_at", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    rel = self.store.get_node(data.get("rel_tag")) if data.get("rel_tag") else None
                    sn, tn = self.store.get_node(src_id), self.store.get_node(dst_id)
                    rows.append((
                        data.get("valid_at", ""), data.get("created_at", ""),
                        FactLine(src=sn.name if sn else src_id,
                                 rel=rel.name if rel else "related_to",
                                 dst=tn.name if tn else dst_id,
                                 valid_at=data.get("valid_at", ""),
                                 invalid_at=data.get("invalid_at", ""),
                                 episode_id=data.get("episode_id", ""))))
        # order by valid-time start, then transaction time; "" (unknown) sorts first
        rows.sort(key=lambda r: (r[0] or "", r[1] or ""))
        return [fl for _, _, fl in rows[:limit]]
