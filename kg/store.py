"""GraphStore — NetworkX (topology) + a node dict + a NumPy vector index, all
persisted to a single SQLite file (docs/ARCHITECTURE.md §7).

NetworkX is a `MultiDiGraph` (directed): structural edges (MENTIONED_IN, RESOLVES_TO,
TAGGED_AS) and FACT edges (RELATED_TO) keep their real direction, and multiple typed /
time-bounded edges can coexist between the same pair. `neighbors()` walks BOTH
directions by default, so traversal stays bidirectional where that matters. PPR/BFS run
over a weighted *undirected* projection built on demand by the retriever
(store-directed, symmetrize-for-diffusion — HippoRAG's PPR runs undirected).

Fact edges are **bi-temporal**: each carries `valid_at` / `invalid_at` (valid-time) and a
`belief` state (transaction-time). Evolution closes a window and opens a new edge rather
than overwriting — `close_facts` / `find_facts` are the helpers the temporal ingest logic
(kg/temporal.py) uses. See docs/TEMPORAL.md.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

import networkx as nx
import numpy as np

from .config import Config
from .models import Belief, Edge, EdgeType, Node, NodeType
from .vectors import VectorIndex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Symmetric edge types have no real direction (embedding cosine / shared-attribute
# overlap). They are pinned to a canonical (min,max) orientation so a pair yields exactly
# ONE row and the undirected projection can't double-count their weight. Symmetric *facts*
# (works_with) are pinned upstream in kg/temporal.py before they reach add_edge.
_SYMMETRIC_ETYPES = {EdgeType.SIMILAR_TO.value, EdgeType.SHARED_TAG.value,
                     EdgeType.SHARED_ENTITY.value}


def fact_active(data: dict, as_of: str | None) -> bool:
    """Is a RELATED_TO fact edge active for the requested view?

    `as_of=None` → the CURRENT view: believed and still open (`invalid_at == ""` = ∞).
    `as_of=T`    → the AS-OF-T view: believed and T inside the valid window. Timestamps
    are ISO strings, so lexical comparison is chronological (a bare year like "2022" also
    compares correctly against full ISO stamps)."""
    if data.get("belief", Belief.ASSERTED.value) != Belief.ASSERTED.value:
        return False
    val, inv = data.get("valid_at", ""), data.get("invalid_at", "")
    if as_of is None:
        return inv == ""
    if val and val > as_of:
        return False
    if inv and inv <= as_of:
        return False
    return True


class GraphStore:
    def __init__(self, config: Config | None = None, path: str | None = None):
        self.config = config or Config.default()
        self.path = path
        self.g = nx.MultiDiGraph()
        self.nodes: dict[str, Node] = {}
        self.vectors = VectorIndex(self.config.embed_dim)
        self.hash_cache: dict[str, str] = {}  # content_hash -> episode node id
        self._ep_count: int | None = None     # cached len(EPISODE nodes), see episode_count()
        # Mutation bookkeeping: save() persists only what changed (write-through), and
        # retrieval-side caches (projection / BM25) key off `version` to know when the
        # graph has moved under them. Every mutator below bumps _touch().
        # `episode_version` moves only when the EPISODE set changes — the BM25 corpus
        # (immutable episode text) depends on nothing else.
        self.version: int = 0
        self.episode_version: int = 0
        self._loading = False                 # _load() replays rows; don't mark them dirty
        self._dirty_nodes: set[str] = set()
        self._dirty_edges: set[tuple[str, str, str]] = set()   # (src, dst, gkey)
        self._dirty_vectors: set[tuple[str, str]] = set()      # (kind, node_id)
        self._dirty_cache: set[str] = set()                    # content hashes
        self._deleted_nodes: set[str] = set()
        self._deleted_edge_rows: set[tuple] = set()   # full SQL PKs to DELETE (see touch_edge)
        self.vectors.on_add = self._mark_vector

    def _mark_vector(self, kind: str, node_id: str) -> None:
        if not self._loading:
            self._dirty_vectors.add((kind, node_id))

    def add_hash(self, content_hash: str, node_id: str) -> None:
        self.hash_cache[content_hash] = node_id
        self._dirty_cache.add(content_hash)

    def _touch(self) -> None:
        if not self._loading:
            self.version += 1

    # ------------------------------------------------------------------ nodes
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.g.add_node(node.id, ntype=node.ntype.value)
        if node.ntype == NodeType.EPISODE:
            self._ep_count = None  # invalidate cache
            if not self._loading:
                self.episode_version += 1
        self._dirty_nodes.add(node.id)
        self._deleted_nodes.discard(node.id)
        self._touch()

    def touch_node(self, node_id: str) -> None:
        """Mark an existing node's payload as changed (aliases, doc_frequency, …) so the
        next flush persists it. In-place mutators MUST call this — save() no longer
        rewrites the world, it only writes what was touched."""
        self._dirty_nodes.add(node_id)
        self._touch()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its incident edges (communities rebuild path). Tracked so the
        next flush deletes its rows; incident edge rows go with it (same DELETE)."""
        if node_id in self.g:
            self.g.remove_node(node_id)
        self.nodes.pop(node_id, None)
        self._dirty_nodes.discard(node_id)
        self._deleted_nodes.add(node_id)
        self._ep_count = None
        self.episode_version += 1
        self._touch()

    def episode_count(self) -> int:
        """Cached count of valid EPISODE nodes (the IDF denominator); lazy."""
        if self._ep_count is None:
            self._ep_count = sum(1 for n in self.nodes.values()
                                 if n.ntype == NodeType.EPISODE and n.valid)
        return self._ep_count

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def nodes_of_type(self, ntype: NodeType, valid_only: bool = True) -> list[Node]:
        return [n for n in self.nodes.values()
                if n.ntype == ntype and (n.valid or not valid_only)]

    # ------------------------------------------------------------------ edges
    def add_edge(self, edge: Edge) -> None:
        if edge.src == edge.dst:
            return
        if not edge.created_at:
            edge.created_at = now_iso()
        etype, rel, disc = edge.key()
        if etype in _SYMMETRIC_ETYPES and edge.src > edge.dst:  # pin canonical orientation
            edge.src, edge.dst = edge.dst, edge.src
        gkey = f"{etype}:{rel}:{disc}"
        # collapse a duplicate of the SAME edge identity, keeping the stronger one.
        existing = self.g.get_edge_data(edge.src, edge.dst)
        if existing and gkey in existing:
            data = existing[gkey]
            if edge.confidence * edge.weight < data["confidence"] * data["weight"]:
                return  # an equal-or-stronger edge already exists
            self.g.remove_edge(edge.src, edge.dst, key=gkey)
        self.g.add_edge(
            edge.src, edge.dst, key=gkey, etype=etype, rel_tag=edge.rel_tag,
            provenance=edge.provenance.value, confidence=edge.confidence,
            weight=edge.weight, valid_at=edge.valid_at, invalid_at=edge.invalid_at,
            belief=edge.belief.value if isinstance(edge.belief, Belief) else edge.belief,
            episode_id=edge.episode_id, valid=edge.valid, created_at=edge.created_at,
        )
        self._dirty_edges.add((edge.src, edge.dst, gkey))
        self._touch()

    def touch_edge(self, src: str, dst: str, gkey: str,
                   old_valid_at: str | None = None) -> None:
        """Mark an edge whose attribute dict was mutated in place (kg/temporal.py's
        confirm/supersede/backfill) so the next flush persists it.

        `old_valid_at` MUST be passed when the mutation changed the edge's `valid_at`
        (backfill / confirm filling an unknown start): valid_at is part of the SQL
        primary key, so the row under the old key has to be deleted or it would come
        back as a duplicate fact on the next load."""
        if old_valid_at is not None:
            d = (self.g.get_edge_data(src, dst) or {}).get(gkey)
            if d is not None and old_valid_at != d.get("valid_at", ""):
                self._deleted_edge_rows.add(
                    (src, dst, d["etype"], d.get("rel_tag") or "", old_valid_at))
        self._dirty_edges.add((src, dst, gkey))
        self._touch()

    # ----------------------------------------------------- fact-edge helpers
    def find_facts(self, src: str, dst: str | None = None, rel_tag: str | None = None,
                   open_only: bool = False):
        """Yield (dst, gkey, data) for RELATED_TO fact edges out of `src`, optionally
        filtered by target / predicate / still-open (`invalid_at == ""`)."""
        if src not in self.g:
            return
        for v, edges in self.g.succ[src].items():
            if dst is not None and v != dst:
                continue
            for gkey, data in edges.items():
                if data.get("etype") != EdgeType.RELATED_TO.value:
                    continue
                if rel_tag is not None and data.get("rel_tag") != rel_tag:
                    continue
                if open_only and data.get("invalid_at", ""):
                    continue
                yield v, gkey, data

    def close_facts(self, src: str, dst: str, rel_tag: str, at: str) -> int:
        """Close every still-open (src→dst, rel_tag) fact at time `at` (set invalid_at).
        Returns the number closed. The edge stays in the graph — just no longer current."""
        closed = 0
        for _v, gkey, data in self.find_facts(src, dst, rel_tag, open_only=True):
            data["invalid_at"] = at
            self.touch_edge(src, dst, gkey)
            closed += 1
        return closed

    def edge_rel_tags(self, src: str, dst: str,
                      etype: EdgeType = EdgeType.RELATED_TO) -> list[str]:
        """Relationship-tag ids on the directed src→dst connection (across parallel edges)."""
        data = self.g.get_edge_data(src, dst)
        if not data:
            return []
        return [d["rel_tag"] for d in data.values()
                if d.get("etype") == etype.value and d.get("rel_tag")]

    def neighbors(self, node_id: str, etypes: set[EdgeType] | None = None,
                  valid_only: bool = True, direction: str = "both"):
        """Yield (neighbor_id, edge_data) over the directed graph.

        `direction`: "both" (default — successors *and* predecessors, so traversal is
        bidirectional regardless of stored edge direction), "out" (successors only), or
        "in" (predecessors only)."""
        if node_id not in self.g:
            return
        want = {e.value for e in etypes} if etypes else None
        adjacencies = []
        if direction in ("out", "both"):
            adjacencies.append(self.g.succ[node_id])
        if direction in ("in", "both"):
            adjacencies.append(self.g.pred[node_id])
        for adj in adjacencies:
            for nbr, edges in adj.items():
                for _k, data in edges.items():
                    if valid_only and not data.get("valid", True):
                        continue
                    if want and data.get("etype") not in want:
                        continue
                    yield nbr, data

    def all_edges(self):
        for u, v, data in self.g.edges(data=True):
            yield u, v, data

    # --------------------------------------------------------------- helpers
    def episode_of(self, mention_id: str) -> str | None:
        n = self.nodes.get(mention_id)
        return n.episode_id if n and n.ntype == NodeType.MENTION else None

    def entity_episodes(self, entity_id: str) -> set[str]:
        """Episodes that reference an entity, via its mention star
        (entity ← RESOLVES_TO ← mention → MENTIONED_IN → episode)."""
        out: set[str] = set()
        for nbr, data in self.neighbors(entity_id, etypes={EdgeType.RESOLVES_TO},
                                        direction="in"):
            ep = self.episode_of(nbr)
            if ep:
                out.add(ep)
        return out

    # ------------------------------------------------------------ persistence
    @classmethod
    def open(cls, path: str, config: Config | None = None) -> "GraphStore":
        store = cls(config=config, path=path)
        if os.path.exists(path):
            store._load()
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            store._init_db()
        return store

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        # WAL: a crash mid-flush can't corrupt the store, and readers don't block the
        # writer. NORMAL sync is the standard WAL pairing (durable to app crash; an OS
        # crash can lose the last transaction, never corrupt).
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def _init_db(self) -> None:
        con = self._connect()
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes(
                id TEXT PRIMARY KEY, ntype TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS edges(
                src TEXT, dst TEXT, etype TEXT, rel_tag TEXT, provenance TEXT,
                confidence REAL, weight REAL, valid_at TEXT, invalid_at TEXT,
                belief TEXT, episode_id TEXT, valid INTEGER, created_at TEXT,
                PRIMARY KEY (src, dst, etype, rel_tag, valid_at));
            CREATE TABLE IF NOT EXISTS vectors(
                node_id TEXT, kind TEXT, vec BLOB, PRIMARY KEY (node_id, kind));
            CREATE TABLE IF NOT EXISTS cache(
                content_hash TEXT PRIMARY KEY, node_id TEXT, ingested_at TEXT);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            """
        )
        con.commit()
        con.close()

    def save(self) -> None:
        """Flush every change since the last save/flush to SQLite (write-through model:
        one transaction of upserts/deletes for exactly the dirty items, NOT a full
        rewrite — save cost is proportional to what changed, so the ingest loop can
        flush periodically and a crash loses at most one flush window)."""
        if not self.path:
            raise ValueError("GraphStore has no path; open(path) first")
        if not (self._dirty_nodes or self._dirty_edges or self._dirty_vectors
                or self._dirty_cache or self._deleted_nodes or self._deleted_edge_rows):
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._init_db()
        con = self._connect()
        cur = con.cursor()
        if self._deleted_nodes:
            gone = [(nid,) for nid in self._deleted_nodes]
            cur.executemany("DELETE FROM nodes WHERE id=?", gone)
            cur.executemany("DELETE FROM vectors WHERE node_id=?", gone)
            cur.executemany("DELETE FROM edges WHERE src=? OR dst=?",
                            [(nid, nid) for nid in self._deleted_nodes])
        if self._deleted_edge_rows:
            cur.executemany(
                "DELETE FROM edges WHERE src=? AND dst=? AND etype=? AND rel_tag=? "
                "AND valid_at=?", sorted(self._deleted_edge_rows))
        cur.executemany(
            "INSERT OR REPLACE INTO nodes(id, ntype, payload) VALUES (?,?,?)",
            [(n.id, n.ntype.value, n.to_payload())
             for nid in sorted(self._dirty_nodes) if (n := self.nodes.get(nid))],
        )
        edge_rows = []
        for u, v, gkey in sorted(self._dirty_edges):
            d = (self.g.get_edge_data(u, v) or {}).get(gkey)
            if d is None:
                continue   # replaced-then-removed within the window; the survivor is dirty too
            edge_rows.append((u, v, d["etype"], d.get("rel_tag") or "", d["provenance"],
                              d["confidence"], d["weight"], d.get("valid_at", ""),
                              d.get("invalid_at", ""), d.get("belief", "asserted"),
                              d.get("episode_id", ""), int(d.get("valid", True)),
                              d.get("created_at", "")))
        cur.executemany(
            "INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", edge_rows)
        vec_rows = []
        for kind, nid in sorted(self._dirty_vectors):
            vec = self.vectors.get(kind, nid)
            if vec is not None:
                vec_rows.append((nid, kind, np.asarray(vec, dtype=np.float32).tobytes()))
        cur.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?)", vec_rows)
        cur.executemany(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?)",
            [(h, self.hash_cache[h], "")
             for h in sorted(self._dirty_cache) if h in self.hash_cache],
        )
        con.commit()
        con.close()
        self._dirty_nodes.clear()
        self._dirty_edges.clear()
        self._dirty_vectors.clear()
        self._dirty_cache.clear()
        self._deleted_nodes.clear()
        self._deleted_edge_rows.clear()

    flush = save   # the ingest loop's periodic checkpoint is the same operation

    def _load(self) -> None:
        self._loading = True
        try:
            self._load_rows()
        finally:
            self._loading = False

    def _load_rows(self) -> None:
        con = self._connect()
        cur = con.cursor()
        for _id, _ntype, payload in cur.execute("SELECT id, ntype, payload FROM nodes"):
            node = Node.from_payload(payload)
            self.nodes[node.id] = node
            self.g.add_node(node.id, ntype=node.ntype.value)
        for row in cur.execute("SELECT * FROM edges"):
            (src, dst, etype, rel_tag, prov, conf, weight, valid_at, invalid_at,
             belief, episode_id, valid, created) = row
            rel_tag = rel_tag or None
            disc = valid_at if etype == EdgeType.RELATED_TO.value else ""
            self.g.add_edge(
                src, dst, key=f"{etype}:{rel_tag or ''}:{disc}", etype=etype,
                rel_tag=rel_tag, provenance=prov, confidence=conf, weight=weight,
                valid_at=valid_at, invalid_at=invalid_at, belief=belief,
                episode_id=episode_id, valid=bool(valid), created_at=created)
        for node_id, kind, blob in cur.execute("SELECT node_id, kind, vec FROM vectors"):
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size:
                self.vectors.dim = vec.size
                self.vectors.add(kind, node_id, vec)
        for content_hash, node_id, _ in cur.execute("SELECT * FROM cache"):
            self.hash_cache[content_hash] = node_id
        con.close()

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for n in self.nodes.values():
            by_type[n.ntype.value] = by_type.get(n.ntype.value, 0) + 1
        by_edge: dict[str, int] = {}
        open_facts = closed_facts = 0
        for _u, _v, d in self.g.edges(data=True):
            by_edge[d["etype"]] = by_edge.get(d["etype"], 0) + 1
            if d["etype"] == EdgeType.RELATED_TO.value:
                if d.get("invalid_at", ""):
                    closed_facts += 1
                else:
                    open_facts += 1
        return {
            "nodes": len(self.nodes),
            "edges": self.g.number_of_edges(),
            "by_node_type": by_type,
            "by_edge_type": by_edge,
            "facts": {"open": open_facts, "closed": closed_facts},
        }
