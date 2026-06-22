"""GraphStore — NetworkX (topology) + a node dict + a NumPy vector index, all
persisted to a single SQLite file (docs/ARCHITECTURE.md §7).

NetworkX is a `MultiDiGraph` (directed, §2 rev 3): edges store their real
direction (src→dst) so relationship semantics survive ("manages", "founded",
"located_in"), and multiple typed edges (e.g. SHARED_TAG *and* SIMILAR_TO) can
coexist between the same pair. `neighbors()` still walks BOTH directions by
default, so traversal stays bidirectional where that matters. PPR/BFS run over a
weighted *undirected* simple-graph projection built on demand by the retriever
(store-directed, symmetrize-for-diffusion — keeps HippoRAG-style PPR recall while
the stored graph remains directional).
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

import networkx as nx
import numpy as np

from .config import Config
from .models import Edge, EdgeType, Node, NodeType
from .vectors import VectorIndex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Symmetric edge types have no real direction (embedding cosine / shared-attribute
# overlap). On the directed store they're pinned to a canonical (min,max) orientation
# so a pair yields exactly ONE row — otherwise the kNN pass would add both a→b and
# b→a and the undirected traversal projection would double-count their weight.
_SYMMETRIC_ETYPES = {EdgeType.SIMILAR_TO.value, EdgeType.SHARED_TAG.value,
                     EdgeType.SHARED_ENTITY.value}


class GraphStore:
    def __init__(self, config: Config | None = None, path: str | None = None):
        self.config = config or Config.default()
        self.path = path
        self.g = nx.MultiDiGraph()
        self.nodes: dict[str, Node] = {}
        self.vectors = VectorIndex(self.config.embed_dim)
        self.hash_cache: dict[str, str] = {}  # content_hash -> object node id
        self._obj_count: int | None = None    # cached len(OBJECT nodes), see object_count()

    # ------------------------------------------------------------------ nodes
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.g.add_node(node.id, ntype=node.ntype.value)
        if node.ntype == NodeType.OBJECT:
            self._obj_count = None  # invalidate cache

    def object_count(self) -> int:
        """Cached count of *valid* OBJECT nodes (the IDF denominator); lazy."""
        if self._obj_count is None:
            self._obj_count = sum(1 for n in self.nodes.values()
                                  if n.ntype == NodeType.OBJECT and n.valid)
        return self._obj_count

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
        etype, rel = edge.key()
        if etype in _SYMMETRIC_ETYPES and edge.src > edge.dst:  # pin canonical orientation
            edge.src, edge.dst = edge.dst, edge.src
        rel_tags = list(dict.fromkeys(edge.rel_tags or []))
        # collapse a duplicate directed (src→dst, etype, relation): keep the stronger
        # one but UNION the relationship-tag sets so a second mention of the same
        # connection accumulates labels (is_friend_of + works_with) instead of racing.
        existing = self.g.get_edge_data(edge.src, edge.dst)
        if existing:
            for k, data in list(existing.items()):
                if data.get("etype") == etype and data.get("relation", "") == rel:
                    merged = list(dict.fromkeys(list(data.get("rel_tags") or []) + rel_tags))
                    if edge.confidence * edge.weight >= data["confidence"] * data["weight"]:
                        self.g.remove_edge(edge.src, edge.dst, key=k)
                        rel_tags = merged
                        break
                    else:
                        data["rel_tags"] = merged   # absorb labels, keep stronger edge
                        return
        self.g.add_edge(
            edge.src, edge.dst, key=f"{etype}:{rel}",
            etype=etype, relation=rel, provenance=edge.provenance.value,
            confidence=edge.confidence, weight=edge.weight,
            valid=edge.valid, created_at=edge.created_at, rel_tags=rel_tags,
        )

    def edge_rel_tags(self, src: str, dst: str,
                      etype: EdgeType = EdgeType.RELATED_TO) -> list[str]:
        """Relationship-tag ids currently on the directed src→dst edge of `etype`."""
        data = self.g.get_edge_data(src, dst)
        if not data:
            return []
        for _k, d in data.items():
            if d.get("etype") == etype.value:
                return list(d.get("rel_tags") or [])
        return []

    def neighbors(self, node_id: str, etypes: set[EdgeType] | None = None,
                  valid_only: bool = True, direction: str = "both"):
        """Yield (neighbor_id, edge_data) over the directed graph.

        `direction`: "both" (default — successors *and* predecessors, so traversal
        is bidirectional regardless of stored edge direction), "out" (successors
        only), or "in" (predecessors only). The default preserves the pre-rev-3
        bidirectional contract every retriever/derivation relies on; "out"/"in" let
        callers (viz, inspect) honour the real direction of relationship edges.
        """
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

    # ----------------------------------------------------- soft-invalidation
    def supersede_node(self, old_id: str, new_id: str) -> None:
        """Mark a node and its incident edges superseded by `new_id` (§2 rev 2)."""
        old = self.nodes.get(old_id)
        if not old:
            return
        old.valid = False
        old.superseded_by = new_id
        old.last_modified = now_iso()
        self._obj_count = None  # a valid object just became invalid
        # directed graph: walk in- AND out-edges so neither direction is missed
        for _u, _v, data in (list(self.g.in_edges(old_id, data=True))
                             + list(self.g.out_edges(old_id, data=True))):
            data["valid"] = False

    # --------------------------------------------------------------- seen flag
    def clear_seen(self) -> None:
        for n in self.nodes.values():
            n.seen = False

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
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        con = self._connect()
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes(
                id TEXT PRIMARY KEY, ntype TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS edges(
                src TEXT, dst TEXT, etype TEXT, relation TEXT, provenance TEXT,
                confidence REAL, weight REAL, valid INTEGER, created_at TEXT,
                rel_tags TEXT,
                PRIMARY KEY (src, dst, etype, relation));
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
        if not self.path:
            raise ValueError("GraphStore has no path; open(path) first")
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._init_db()
        con = self._connect()
        cur = con.cursor()
        cur.execute("DELETE FROM nodes")
        cur.execute("DELETE FROM edges")
        cur.execute("DELETE FROM vectors")
        cur.executemany(
            "INSERT INTO nodes(id, ntype, payload) VALUES (?,?,?)",
            [(n.id, n.ntype.value, n.to_payload()) for n in self.nodes.values()],
        )
        edge_rows = []
        for u, v, d in self.g.edges(data=True):
            edge_rows.append((u, v, d["etype"], d.get("relation", ""), d["provenance"],
                              d["confidence"], d["weight"], int(d.get("valid", True)),
                              d.get("created_at", ""),
                              json.dumps(d.get("rel_tags") or [])))
        cur.executemany(
            "INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?,?,?,?,?,?)", edge_rows)
        vec_rows = [(nid, kind, np.asarray(vec, dtype=np.float32).tobytes())
                    for kind, nid, vec in self.vectors.iter_vectors()]
        cur.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?)", vec_rows)
        cur.executemany(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?)",
            [(h, nid, "") for h, nid in self.hash_cache.items()],
        )
        con.commit()
        con.close()

    def _load(self) -> None:
        con = self._connect()
        cur = con.cursor()
        for _id, _ntype, payload in cur.execute("SELECT id, ntype, payload FROM nodes"):
            node = Node.from_payload(payload)
            self.nodes[node.id] = node
            self.g.add_node(node.id, ntype=node.ntype.value)
        for row in cur.execute("SELECT * FROM edges"):
            # tolerant of pre-rev-3 stores that have no `rel_tags` column
            src, dst, etype, relation, prov, conf, weight, valid, created = row[:9]
            rel_tags = json.loads(row[9]) if len(row) > 9 and row[9] else []
            self.g.add_edge(
                src, dst, key=f"{etype}:{relation}", etype=etype, relation=relation,
                provenance=prov, confidence=conf, weight=weight,
                valid=bool(valid), created_at=created, rel_tags=rel_tags)
        for node_id, kind, blob in cur.execute("SELECT node_id, kind, vec FROM vectors"):
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size:
                self.vectors.dim = vec.size  # adopt the stored embedding dim
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
        for _u, _v, d in self.g.edges(data=True):
            by_edge[d["etype"]] = by_edge.get(d["etype"], 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": self.g.number_of_edges(),
            "by_node_type": by_type,
            "by_edge_type": by_edge,
        }
