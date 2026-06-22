"""GraphStore — NetworkX (topology) + a node dict + a NumPy vector index, all
persisted to a single SQLite file (docs/ARCHITECTURE.md §7).

NetworkX is a `MultiGraph` (undirected = bidirectional by construction, §2), so a
node's neighbours are returned regardless of stored edge direction, and multiple
typed edges (e.g. SHARED_TAG *and* SIMILAR_TO) can coexist between the same pair.
PPR/BFS run over a weighted simple-graph projection built on demand by the
retriever so it can honour the confidence floor and skip superseded edges.
"""
from __future__ import annotations

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


class GraphStore:
    def __init__(self, config: Config | None = None, path: str | None = None):
        self.config = config or Config.default()
        self.path = path
        self.g = nx.MultiGraph()
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
        # collapse a duplicate (src,dst,etype,relation): keep the stronger one
        existing = self.g.get_edge_data(edge.src, edge.dst)
        if existing:
            for k, data in existing.items():
                if data.get("etype") == etype and data.get("relation", "") == rel:
                    if edge.confidence * edge.weight >= data["confidence"] * data["weight"]:
                        self.g.remove_edge(edge.src, edge.dst, key=k)
                        break
                    else:
                        return
        self.g.add_edge(
            edge.src, edge.dst, key=f"{etype}:{rel}",
            etype=etype, relation=rel, provenance=edge.provenance.value,
            confidence=edge.confidence, weight=edge.weight,
            valid=edge.valid, created_at=edge.created_at,
        )

    def neighbors(self, node_id: str, etypes: set[EdgeType] | None = None,
                  valid_only: bool = True):
        """Yield (neighbor_id, edge_data) over the bidirectional graph."""
        if node_id not in self.g:
            return
        want = {e.value for e in etypes} if etypes else None
        for nbr, edges in self.g.adj[node_id].items():
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
        for _u, _v, data in self.g.edges(old_id, data=True):
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
                              d.get("created_at", "")))
        cur.executemany(
            "INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?,?,?,?,?)", edge_rows)
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
            src, dst, etype, relation, prov, conf, weight, valid, created = row
            self.g.add_edge(
                src, dst, key=f"{etype}:{relation}", etype=etype, relation=relation,
                provenance=prov, confidence=conf, weight=weight,
                valid=bool(valid), created_at=created)
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
