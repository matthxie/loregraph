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
    mentions: int = 1
    last_mentioned: str = ""
    event: bool = False        # dated occurrence: render "on d" / "d1 -> d2", never state grammar

    @classmethod
    def from_edge(cls, store: GraphStore, src_id: str, dst_id: str,
                  data: dict) -> "FactLine":
        """One line from a RELATED_TO edge in stored orientation: endpoint/predicate
        names resolved through the store, structured extras lifted off the edge data."""
        rel = data.get("rel_tag")
        rel_node = store.get_node(rel) if rel else None
        sn, tn = store.get_node(src_id), store.get_node(dst_id)
        confirmed = list(data.get("confirmed_by") or [])
        last_mentioned = ""
        for ep in confirmed:
            ep_node = store.get_node(ep)
            if ep_node is not None and ep_node.created_at > last_mentioned:
                last_mentioned = ep_node.created_at
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
                   disputed_by=list(data.get("disputed_by") or []),
                   mentions=1 + len(confirmed),
                   last_mentioned=last_mentioned,
                   event=bool(data.get("event", False)))

    def to_row(self) -> dict:
        """The wire Fact object (PROTOCOL §3): structured fields plus this line's
        rendered form. Store-empty strings cross as null per the §3 conventions."""
        return {"source": self.src, "predicate": self.rel, "target": self.dst,
                # an event's closed window means "happened", not "ceased to hold"
                "status": ("occurred" if self.event and self.invalid_at
                           else "ended" if self.invalid_at else "asserted"),
                "valid_from": self.valid_at or None,
                "valid_to": self.invalid_at or None,
                "recorded_at": self.recorded_at or None,
                "episode_id": self.episode_id or None,
                "confidence": self.confidence,
                "provenance": (self.provenance or "").lower() or None,
                "functional": self.functional,
                "disputed_by": self.disputed_by or [],
                "mentions": self.mentions,
                "last_mentioned": self.last_mentioned or None,
                "rendered": self.render()}

    def render(self) -> str:
        win = []
        # `valid_at` on an open fact is usually just the asserting session's created_at,
        # not a confirmed real-world start date — "since" overclaims precision the extractor
        # never had. Only a CLOSED window (invalid_at present) is a real bi-temporal interval,
        # so "since/until" is reserved for that; an open fact gets the honest "mentioned".
        # (Deliberate deviation from PROTOCOL §5.2's Rust line grammar; the explicit
        # "ended" marker keeps a closed line unmistakable next to the row's status.)
        if self.event and self.valid_at:
            # OCCURRENCE grammar (edge stamped event=True by kg/temporal.py): the window
            # is when it HAPPENED, so "since/until/ended" state grammar would misread a
            # past event as a lapsed state. Same-day re-mentions confirm-collapse
            # (confirmed_by), so the frequency stays visible; distinct dated occurrences
            # are separate edges → separate lines.
            if self.invalid_at and self.invalid_at[:10] != self.valid_at[:10]:
                win.append(f"{self.valid_at[:10]} -> {self.invalid_at[:10]}")
            else:
                win.append(f"on {self.valid_at[:10]}")
            if self.mentions > 1:
                win.append(f"mentioned {self.mentions}x")
        elif self.valid_at and self.invalid_at:
            win.append(f"since {self.valid_at[:10]}")
            win.append(f"until {self.invalid_at[:10]}")
            win.append("ended")
        elif self.valid_at:
            if self.mentions > 1:
                # repeated undated assertions confirm-collapse into one edge; surface the
                # frequency and the first->latest mention span so the answer LLM sees it
                last = (self.last_mentioned or self.valid_at)[:10]
                win.append(f"mentioned {self.mentions}x ({self.valid_at[:10]} -> {last})")
            else:
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
        # order by valid-time start, then transaction time; "" (unknown) sorts first.
        # CAP POLICY: the pre-fix cap kept the OLDEST `limit` rows (head of the ascending
        # sort), silently cutting every recent closure on a hub with >limit facts in favor
        # of old filler — inverted from what an evolution answer needs. The fix ranks the
        # capped SELECTION by recency, with CLOSED rows kept preferentially: closures are
        # the lines only this block carries (open rows mostly restate the FACTS section),
        # so they must never lose their seat to newer open filler. The kept rows still
        # read in ascending time order.
        rows.sort(key=lambda r: (r[0] or "", r[1] or ""))
        if limit and len(rows) > limit:
            closed = [r for r in rows if r[2].invalid_at][-limit:]
            spare = limit - len(closed)
            open_rows = [r for r in rows if not r[2].invalid_at][-spare:] if spare else []
            rows = sorted(closed + open_rows, key=lambda r: (r[0] or "", r[1] or ""))
        return [fl for _, _, fl in rows[:limit]] if limit else []
