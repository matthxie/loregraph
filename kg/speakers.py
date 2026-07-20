"""Speaker provenance — who said the material a fact rests on (docs/OFFLINE_EVAL.md Round 8).

MOTIVATION. The reader miscomputes answers from figures it should refuse because it cannot
tell WHO stated them: the assistant's generic price ranges / typical fares / example head
counts sit in the same conversations as the user's own stated facts, and a
"what is true of you / what did you spend" question that reads an assistant figure as a
user fact answers with a number that was never the user's (the *_abs abstention failures
031748ae_abs, 19b5f2b3_abs, 09ba9854). Chat chunks already carry inline "User:"/"Assistant:"
turn markers, so speaker is DERIVABLE from raw text with no LLM — a $0 local signal.

DESIGN (docs/OFFLINE_EVAL.md Round 8). Speaker provenance is METADATA you filter by, never
a diffusion signal (no graph edges → no per-speaker super-hub) and never stored on fact
edges (derivable → would go stale on retract/forget/merge/canonical-rename). Two pieces:

  1. A `speakers` REGISTRY table in the store (kg/store.py): one row per speaker
     (speaker_id, kind, canonical_name, aliases[]). Reference data — for now every graph
     has the canonical user/assistant (+ a `mixed` bucket for multi-role chunks); aliases[]
     is forward-scaffolding for multi-human identity resolution (stored, NO resolution logic).
  2. A `speaker_id` FIELD stamped on each immutable EPISODE/chunk node (Node.speaker_id).
     ONE stamp per chunk. "chunks by speaker" is a QUERY (nodes where speaker_id=X), never a
     stored list on the speaker row.

ATTRIBUTION is DERIVED at read time: a fact's speakers = the speaker kinds of its provenance
episodes (edge `episode_id` ∪ `confirmed_by`) resolved through the registry. The reader
marker uses the ANY-USER reduction: a fact is USER-GROUNDED if ANY provenance turn is a human
(a user fact echoed by the assistant is still a user fact); it is marked "[assistant]" only
when it rests EXCLUSIVELY on assistant turns. A MIXED chunk (both roles present — chunking is
not guaranteed single-turn) counts as containing-human (conservative: never wrongly discount
a user fact). An episode with no marker (a plain note, described media) stamps no speaker_id
and contributes no kind — so it can never, on its own, trigger an [assistant] mark either.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .models import Belief, EdgeType, NodeType
from .store import GraphStore

# Role markers at a line start, mapped onto two kinds. Tolerant of leading whitespace/quote
# marks and an optional bold wrapper, matching how chat turns actually render; deliberately a
# superset of chunkers._TURN so a packed multi-turn chunk is still classified. Deterministic,
# no LLM.
_ROLE = re.compile(r"(?im)^[ \t>*]*\**\s*(user|assistant|human|ai)\s*\**\s*:")
_ROLE_KIND = {"user": "human", "human": "human", "assistant": "assistant", "ai": "assistant"}

# kind -> (canonical_name, default aliases). The canonical_name is the stable identity the
# speaker_id hashes; aliases[] are the surface forms (duplicates across ids are expected once
# multi-human lands — stored as reference data, never resolved here).
_KIND_CANON = {"human": "user", "assistant": "assistant", "mixed": "mixed"}
_KIND_ALIASES = {"human": ["User", "Human"], "assistant": ["Assistant", "AI"], "mixed": []}


def _speaker_id(canonical_name: str) -> str:
    """Stable, content-addressed speaker id = a truncated SHA-256 of the canonical name.
    Deterministic across stores/backfills so the same speaker always keys the same row."""
    return "sp_" + hashlib.sha256(canonical_name.encode("utf-8")).hexdigest()[:12]


@dataclass
class SpeakerRow:
    """One registry row. Reference data — never carries a chunk list (that is a QUERY)."""
    speaker_id: str
    kind: str                 # 'human' | 'assistant' | 'mixed'
    canonical_name: str
    aliases: list[str] = field(default_factory=list)


def speaker_row_for(kind: str) -> SpeakerRow:
    """The canonical registry row for a kind (user/assistant/mixed)."""
    cname = _KIND_CANON[kind]
    return SpeakerRow(speaker_id=_speaker_id(cname), kind=kind, canonical_name=cname,
                      aliases=list(_KIND_ALIASES[kind]))


def detect_roles(raw_text: str | None) -> set[str]:
    """The distinct speaker kinds whose turn markers appear in the text ({} when none)."""
    if not raw_text:
        return set()
    return {_ROLE_KIND[m.group(1).lower()] for m in _ROLE.finditer(raw_text)}


def parse_speaker(raw_text: str | None) -> tuple[str | None, str | None]:
    """(speaker_id, kind) for a chunk's raw text, from its inline turn markers. Both roles
    present → kind='mixed' (chunking is not guaranteed single-turn). No marker → (None, None):
    the chunk gets no stamp and contributes no kind to any fact's attribution."""
    roles = detect_roles(raw_text)
    if not roles:
        return None, None
    if len(roles) > 1:
        kind = "mixed"
    else:
        kind = next(iter(roles))
    return _speaker_id(_KIND_CANON[kind]), kind


# --------------------------------------------------------------------- ingest / backfill
def ensure_speaker(store: GraphStore, kind: str) -> str:
    """Upsert the canonical registry row for `kind` and return its speaker_id. Idempotent —
    re-upserting an identical row is a no-op the store won't re-persist."""
    row = speaker_row_for(kind)
    store.upsert_speaker(row)
    return row.speaker_id


def stamp_episode(store: GraphStore, node) -> str | None:
    """Parse `node.raw_text`, stamp `node.speaker_id`, and upsert the registry. Returns the
    stamped speaker_id (None when the text carries no turn marker). Additive and deterministic
    — safe to run at ingest AND as a backfill; consumers stay gated behind config knobs."""
    sid, kind = parse_speaker(node.raw_text)
    node.speaker_id = sid
    if sid is not None:
        ensure_speaker(store, kind)
    return sid


def backfill_speakers(store: GraphStore) -> dict:
    """Stamp speaker_id on every EPISODE node from its raw_text and build the registry —
    idempotent, incremental (only re-touches a node whose stamp actually changes), and $0
    (pure regex, no LLM). Touches only node payloads + the speakers table, so it CANNOT change
    the ingest-cache key (which hashes config/sessions/prompt, never db contents): the cached
    benchmark stores gain speaker provenance in place with no paid re-extraction.

    Returns counts: stamped (nodes now carrying a speaker_id), changed (payloads re-touched
    this pass), unmarked (episodes with no turn structure), speakers (registry row total).
    """
    stamped = changed = unmarked = 0
    for node in store.nodes_of_type(NodeType.EPISODE, valid_only=False):
        sid, kind = parse_speaker(node.raw_text)
        if sid is None:
            unmarked += 1
            if node.speaker_id is not None:           # a prior stamp no longer holds
                node.speaker_id = None
                store.touch_node(node.id)
                changed += 1
            continue
        stamped += 1
        ensure_speaker(store, kind)
        if node.speaker_id != sid:
            node.speaker_id = sid
            store.touch_node(node.id)
            changed += 1
    return {"stamped": stamped, "changed": changed, "unmarked": unmarked,
            "speakers": len(store.speakers)}


# --------------------------------------------------------------------- attribution (read-time)
def provenance_episodes(data: dict) -> set[str]:
    """The episodes a fact rests on: the asserting `episode_id` ∪ every `confirmed_by`
    re-mention. Mirrors kg/fact_vectors._edge_episodes."""
    eps: set[str] = set()
    ep = data.get("episode_id")
    if ep:
        eps.add(ep)
    eps.update(data.get("confirmed_by") or [])
    return eps


def asserted_by(store: GraphStore, data: dict) -> list[str]:
    """The DERIVED list of distinct speaker kinds behind a fact edge (sorted), resolved from
    its provenance episodes' stamped speaker_id through the registry. Derived at read time,
    never stored on the edge. Empty when no provenance episode carries a speaker_id (an
    un-backfilled store, or a fact grounded only in unmarked notes)."""
    kinds: set[str] = set()
    for eid in provenance_episodes(data):
        n = store.get_node(eid)
        sid = getattr(n, "speaker_id", None) if n else None
        if not sid:
            continue
        row = store.get_speaker(sid)
        if row:
            kinds.add(row.kind)
    return sorted(kinds)


def is_assistant_only(kinds) -> bool:
    """The reader marker condition: mark [assistant] ONLY when there is at least one known
    speaker AND every one is the assistant. Any human/mixed turn (user-grounded) blocks the
    mark; an empty set (unknown provenance) blocks it too — conservative, never discount."""
    kinds = list(kinds)
    return bool(kinds) and all(k == "assistant" for k in kinds)


def assistant_marker(kinds) -> str:
    """The rendered marker suffix ("" when user-grounded/unknown, per is_assistant_only)."""
    return " [assistant]" if is_assistant_only(kinds) else ""
