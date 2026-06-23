"""Agentic graph-traversal query (docs/ARCHITECTURE.md §5 — the reserved
"LLM-guided traversal" path).

§5 reserves live LLM traversal as the *explainability / path-finding* path: "serialize
the current subgraph compactly into the prompt … respect the ~80-node-per-prompt
ceiling." This module realizes it. An LLM answers a natural-language prompt by calling
**read-only graph tools** across turns — seed-and-spread (PPR), keyword/vector search,
typed neighbor expansion, shortest-path, read-object, theme browsing — gathers compact
evidence, then calls `submit_answer` with prose + the object ids it used as citations.

It mirrors the extractor's real⇄offline contract exactly (`HaikuExtractor` ⇄
`HeuristicExtractor`, `get_extractor`): a `ClaudeAgent` (Anthropic tool-use loop) that
degrades to a deterministic `OfflineAgent` running the *same* tool executors, so the
whole thing works with **no API key and no network** and tests stay deterministic.

The agent is pure orchestration over the existing primitives (`PPRRetriever`,
`VectorRetriever`, `Seeder`, `projected_graph`, `store.neighbors`, `CommunityRetriever`,
`Canonicalizer`) — it reimplements no retrieval or seeding.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import networkx as nx

from .canonicalize import Canonicalizer
from .communities import CommunityRetriever, is_global_query
from .config import Config
from .embedders import Embedder
from .models import EdgeType, Node, NodeType
from .retrieval import (PPRRetriever, VectorRetriever, projected_graph)
from .store import GraphStore

# Per-result caps so any single tool stays small even before the node budget bites.
AGENT_MAX_SNIPPET = 160      # chars per object snippet
AGENT_MAX_HITS = 12          # rows any single search/neighbor tool may return
_WS = re.compile(r"\s+")
_OBJ_ID = re.compile(r"\bobj_[A-Za-z0-9_]+\b")
_CONNECT_CUE = re.compile(
    r"\b(connect|connected|relate|related|link|linked|between|relationship|"
    r"how (?:are|is|do|does|did))\b", re.I)
# Some models (notably Haiku) occasionally emit a tool call as LITERAL TEXT rather than a
# real tool_use block ('<submit_answer>\n<parameter name="answer">...'); the prose-salvage
# path unwraps that so the user never sees raw tool-call markup as the answer.
_MARKUP_ANSWER = re.compile(
    r'<parameter name="answer">(.*?)(?:</parameter>|<parameter name=|</submit_answer>|$)',
    re.DOTALL)
_MARKUP_CITES = re.compile(
    r'<parameter name="citations">(.*?)(?:</parameter>|</submit_answer>|$)', re.DOTALL)

# Etypes the agent's neighbors tool exposes (IN_COMMUNITY is excluded — community hubs
# are browsed via browse_themes, not walked, matching retrieval._TRAVERSAL_EXCLUDE).
_NEIGHBOR_ETYPES = ({e.value for e in EdgeType}
                    - {EdgeType.IN_COMMUNITY.value,
                       EdgeType.TAGGED_AS.value, EdgeType.SHARED_TAG.value})  # tags retired


# --------------------------------------------------------------------------- #
# Tool schemas (Anthropic format, same shape as extractors.GRAPH_TOOL)
# --------------------------------------------------------------------------- #
TOOLS: list[dict] = [
    {
        "name": "seed_and_spread",
        "description": "Find the objects most relevant to a sub-question by seeding from "
        "text+keyword+entity matches and diffusing over the graph (Personalized PageRank). "
        "Your PRIMARY retrieval tool — call it first. Returns compact object stubs; call "
        "read_object to see full text before citing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "a focused sub-question or phrase"},
                "k": {"type": "integer", "description": "objects to return (default 8)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "keyword_search",
        "description": "Exact/lexical search over raw object text (BM25). Use for names, "
        "rare terms, or exact phrases an embedding blurs (e.g. 'Enigma machine').",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "vector_search",
        "description": "Flat semantic similarity over object embeddings, no graph. Use ONLY "
        "to broaden recall when seed_and_spread returned too little.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "neighbors",
        "description": "List what a node connects to, with the relationship LABEL, DIRECTION "
        "(out = node→X, in = X→node), provenance and confidence on each edge. Use to verify "
        "or walk a SPECIFIC labeled relationship one hop. For broad relevance prefer "
        "seed_and_spread. Pass an id returned by a search tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "direction": {"type": "string", "enum": ["both", "out", "in"]},
                "etypes": {"type": "array", "items": {
                    "type": "string",
                    "enum": ["MENTIONS", "RELATED_TO", "SIMILAR_TO",
                             "SHARED_ENTITY", "HYPERLINKS_TO"]}},
                "limit": {"type": "integer", "description": "default 20"},
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "find_path",
        "description": "Find the shortest chain of relationships connecting two entities or "
        "objects, to explain HOW they are related. Use only for explicit 'how are X and Y "
        "connected' questions. Pass ids returned by a search tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "target_id": {"type": "string"},
                "max_hops": {"type": "integer", "description": "default 4"},
            },
            "required": ["source_id", "target_id"],
        },
    },
    {
        "name": "read_object",
        "description": "Read an object's FULL text (or an image's description) so you can "
        "quote and cite it. The ONLY tool that returns full content — call it before you "
        "cite an object.",
        "input_schema": {
            "type": "object",
            "properties": {"object_id": {"type": "string"}, "max_chars": {"type": "integer"}},
            "required": ["object_id"],
        },
    },
    {
        "name": "browse_themes",
        "description": "List the corpus-wide themes (community summaries) relevant to a "
        "query. Use for broad 'what are the main topics / overall themes' questions, not "
        "specific facts.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"],
        },
    },
]

SUBMIT_ANSWER_TOOL: dict = {
    "name": "submit_answer",
    "description": "Submit your final answer. Call this exactly once, when you can support "
    "the answer from objects you have read. citations = the object ids you used as evidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"},
                          "description": "object ids you read and used, e.g. obj_0003"},
        },
        "required": ["answer", "citations"],
    },
}

_AGENT_SYS = (
    "You answer questions using a DIRECTED knowledge graph over a fixed corpus. "
    "Nodes are objects (documents), entities & concepts (named things AND topical themes, "
    "including dates), relations and themes. Objects connect to entities/concepts (MENTIONS); "
    "entities connect to entities "
    "through directed, LABELED relationships (RELATED_TO, e.g. 'founded', 'located_in') "
    "carrying provenance and confidence. You cannot see the graph directly — you explore it "
    "with tools.\n\n"
    "STRATEGY. Start with seed_and_spread on a focused rephrasing of the question; it runs "
    "seed-and-spread (PPR) and is your primary tool. Use keyword_search for exact names or "
    "rare terms; use vector_search ONLY to broaden when results are thin. Use neighbors to "
    "check a SPECIFIC labeled relationship one hop out; use find_path only for explicit "
    "'how are X and Y connected' questions; use browse_themes for broad 'what are the main "
    "themes' questions. Pass the ids a search tool returns into neighbors/find_path.\n\n"
    "DIRECTION & LABELS matter. An edge 'A --founded--> B' does NOT mean B founded A. Honor "
    "the direction and rel fields. Prefer higher-confidence EXTRACTED edges over "
    "low-confidence INFERRED ones.\n\n"
    "CITATION DISCIPLINE. Read an object's full text with read_object before you rely on or "
    "cite it. Cite an object id ONLY after you have read it. Never cite an id no tool "
    "returned. Every factual claim must trace to a cited object; if the evidence is thin, "
    "say so rather than inventing facts.\n\n"
    "STOP as soon as you can answer, or when a tool reports ceiling_hit. Do not re-call a "
    "tool with the same arguments. When ready, call submit_answer with your prose answer and "
    "the list of object ids you used."
)


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #
@dataclass
class AgentAnswer:
    """The agentic analogue of RetrievalResult. `object_ids` (citations first, then any
    other object the agent surfaced) is the eval seam — evaluate.py reads it exactly like
    a RetrievalResult, so recall@k/MRR compare the agent apples-to-apples with PPR/BFS."""
    query: str
    answer: str
    citations: list[str] = field(default_factory=list)        # validated object ids
    dropped_citations: list[str] = field(default_factory=list)  # cited but unvalidated (audit)
    backend: str = "offline"                                   # "claude" | "offline"
    steps: int = 0
    stopped: str = "answered"                                 # answered | prose | step_cap
    trace: list[dict] = field(default_factory=list)           # [{step,tool,input,result_summary}]
    seeds: list[str] = field(default_factory=list)
    touched: list[str] = field(default_factory=list)          # every node a tool surfaced
    object_ids: list[str] = field(default_factory=list)       # ranked object ids (eval seam)
    notes: list[str] = field(default_factory=list)

    def objects(self, store: GraphStore) -> list[Node]:
        """Resolve the validated citations to their ObjectNodes (convenience)."""
        return [n for n in (store.get_node(c) for c in self.citations) if n is not None]


# --------------------------------------------------------------------------- #
# Compact serialization helpers
# --------------------------------------------------------------------------- #
def _snippet(node: Node, n: int = AGENT_MAX_SNIPPET) -> str:
    text = node.raw_text or node.description or node.summary or node.name or ""
    return _WS.sub(" ", text).strip()[:n]


# --------------------------------------------------------------------------- #
# Trace accumulator — feeds AgentAnswer + the viewer schema (viz.agent_trace_payload)
# --------------------------------------------------------------------------- #
class Trace:
    def __init__(self):
        self.steps: list[dict] = []
        self.seeds: list[str] = []     # ordered-unique seed ids
        self.results: list[str] = []   # ordered-unique object ids appearing in ranked lists
        self.read: set[str] = set()    # object ids the model actually read (citation gate)

    @staticmethod
    def _extend_unique(dst: list[str], ids) -> None:
        have = set(dst)
        for i in ids:
            if i not in have:
                have.add(i)
                dst.append(i)

    def record(self, step: int, tool: str, inp: dict, out: dict) -> None:
        if isinstance(out, dict):
            if out.get("seeds"):
                self._extend_unique(self.seeds, [s["id"] for s in out["seeds"]])
            if out.get("objects"):
                self._extend_unique(self.results, [o["id"] for o in out["objects"]])
            if tool == "read_object" and out.get("id") and not out.get("error"):
                self.read.add(out["id"])
        self.steps.append({"step": step, "tool": tool, "input": inp,
                           "result_summary": _summarize_result(tool, out)})


def _summarize_result(tool: str, out: dict) -> str:
    if not isinstance(out, dict):
        return str(out)[:120]
    if out.get("error"):
        return f"error: {out['error']}"
    if out.get("note"):
        return "(deduped)"
    if tool in ("seed_and_spread", "keyword_search", "vector_search"):
        s = f"{len(out.get('objects', []))} objects"
        if out.get("seeds"):
            s += f", {len(out['seeds'])} seeds"
        if out.get("ceiling_hit") or out.get("truncated"):
            s += " (truncated)"
        return s
    if tool == "neighbors":
        return f"{len(out.get('edges', []))} edges" + (" (truncated)" if out.get("truncated") else "")
    if tool == "find_path":
        return f"path found, {out.get('hops')} hops" if out.get("found") else "no path"
    if tool == "read_object":
        return f"{len(out.get('text', '') or '')} chars"
    if tool == "browse_themes":
        return f"{len(out.get('themes', []))} themes"
    return "ok"


# --------------------------------------------------------------------------- #
# GraphTools — the one executor that owns the reused primitives + the node budget
# --------------------------------------------------------------------------- #
@dataclass
class GraphTools:
    store: GraphStore
    embedder: Embedder
    canon: Canonicalizer
    config: Config

    def __post_init__(self):
        self.ppr = PPRRetriever(self.store, self.embedder, self.canon, self.config)
        self.vector = VectorRetriever(self.store, self.embedder, self.canon, self.config)
        self.seeder = self.ppr.seeder            # reuse the SAME Seeder (cached BM25)
        self.community = CommunityRetriever(self.store, self.embedder, self.config)
        self._proj: nx.Graph | None = None       # lazy projected_graph cache
        self.touched: set[str] = set()            # the §5 node budget
        self._order: list[str] = []               # insertion order of touched (ranking)

    # ---- budget -------------------------------------------------------------
    def _budget_left(self) -> int:
        return self.config.agent_node_budget - len(self.touched)

    def _touch(self, node_id: str) -> None:
        if node_id not in self.touched:
            self.touched.add(node_id)
            self._order.append(node_id)

    def touched_ids(self) -> list[str]:
        return list(self._order)

    def projection(self) -> nx.Graph:
        if self._proj is None:
            self._proj = projected_graph(self.store, self.config)
        return self._proj

    # ---- stubs --------------------------------------------------------------
    def _stub(self, node: Node, snippet: bool = False) -> dict:
        d = {"id": node.id, "name": (node.name or node.id)[:80], "ntype": node.ntype.value,
             "modality": node.modality.value if node.modality else None}
        if snippet and node.ntype in (NodeType.OBJECT, NodeType.COMMUNITY):
            d["snippet"] = _snippet(node)
        return d

    def _stub_id(self, node_id: str, snippet: bool = False) -> dict | None:
        n = self.store.get_node(node_id)
        return self._stub(n, snippet) if n else None

    def _rel_label(self, data: dict) -> str:
        """Relationship name for an edge: rel_tag → relation node name; else the legacy
        coarse class; else the edge type. Same resolution as cmd_inspect / viz."""
        rid = data.get("rel_tag")
        if rid:
            rn = self.store.get_node(rid)
            if rn:
                return rn.name
        return data.get("relation") or data.get("etype") or "related_to"

    # ---- dispatch -----------------------------------------------------------
    def dispatch(self, name: str, inp: dict) -> dict:
        inp = inp or {}
        try:
            if name == "seed_and_spread":
                return self._tool_seed_and_spread(inp.get("query", ""), inp.get("k"))
            if name == "keyword_search":
                return self._tool_keyword_search(inp.get("query", ""), inp.get("k"))
            if name == "vector_search":
                return self._tool_vector_search(inp.get("query", ""), inp.get("k"))
            if name == "neighbors":
                return self._tool_neighbors(inp.get("node_id", ""), inp.get("direction", "both"),
                                            inp.get("etypes"), inp.get("limit", 20))
            if name == "find_path":
                return self._tool_find_path(inp.get("source_id", ""), inp.get("target_id", ""),
                                            inp.get("max_hops", 4))
            if name == "read_object":
                return self._tool_read_object(inp.get("object_id", ""), inp.get("max_chars"))
            if name == "browse_themes":
                return self._tool_browse_themes(inp.get("query", ""), inp.get("k", 5))
            return {"error": f"unknown tool {name!r}"}
        except Exception as e:  # noqa: BLE001 — a tool bug must not crash the loop
            return {"error": f"{name} failed: {e!r}"}

    # ---- ranked-object tools (seed_and_spread / keyword / vector) -----------
    def _clamp_k(self, k) -> int:
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = self.config.top_k
        return max(1, min(k, AGENT_MAX_HITS))

    def _object_rows(self, ranked: list[tuple[str, float]]) -> tuple[list[dict], bool]:
        """Turn [(object_id, score)] into compact stubs, registering each into the node
        budget; stop and flag truncation once the §5 ceiling is hit."""
        rows, truncated = [], False
        for oid, score in ranked:
            n = self.store.get_node(oid)
            if n is None:
                continue
            if oid not in self.touched and self._budget_left() <= 0:
                truncated = True
                break
            self._touch(oid)
            row = self._stub(n, snippet=True)
            row["score"] = round(float(score), 4)
            rows.append(row)
        return rows, truncated

    def _tool_seed_and_spread(self, query: str, k=None) -> dict:
        res = self.ppr.retrieve(query, k=self._clamp_k(k))
        # seeds participate in the SAME node budget as objects (they're usually
        # high-value entity/tag hubs, so they're counted first): stop and flag
        # truncation once the §5 ceiling is hit, exactly like _object_rows.
        seeds, seed_trunc = [], False
        for sid in res.seeds[:8]:
            if sid not in self.touched and self._budget_left() <= 0:
                seed_trunc = True
                break
            st = self._stub_id(sid)
            if st:
                self._touch(sid)
                seeds.append(st)
        objects, obj_trunc = self._object_rows(res.objects)
        truncated = seed_trunc or obj_trunc
        return {"seeds": seeds, "objects": objects, "ceiling_hit": truncated,
                "truncated": truncated}

    def _tool_keyword_search(self, query: str, k=None) -> dict:
        ranked = self.seeder.bm25_search(query, k=self._clamp_k(k))
        objects, truncated = self._object_rows(ranked)
        return {"objects": objects, "ceiling_hit": truncated, "truncated": truncated}

    def _tool_vector_search(self, query: str, k=None) -> dict:
        res = self.vector.retrieve(query, k=self._clamp_k(k))
        objects, truncated = self._object_rows(res.objects)
        return {"objects": objects, "ceiling_hit": truncated, "truncated": truncated}

    # ---- neighbors (preserves label + direction + provenance + confidence) --
    def _tool_neighbors(self, node_id: str, direction="both", etypes=None, limit=20) -> dict:
        node = self.store.get_node(node_id)
        if node is None:
            return {"error": f"no such node: {node_id}"}
        direction = direction if direction in ("both", "out", "in") else "both"
        want = None
        if etypes:
            want = {e for e in etypes if e in _NEIGHBOR_ETYPES}
        try:
            limit = max(1, min(int(limit), AGENT_MAX_HITS))
        except (TypeError, ValueError):
            limit = AGENT_MAX_HITS
        # which incident edges are OUT (this node is the source) — same trick as cmd_inspect
        out_ids = {id(d) for _n, d in self.store.neighbors(node_id, direction="out")}
        edges, truncated = [], False
        for nbr, data in self.store.neighbors(node_id):
            if data.get("etype") == EdgeType.IN_COMMUNITY.value:
                continue
            if want is not None and data.get("etype") not in want:
                continue
            nn = self.store.get_node(nbr)
            if nn is None:
                continue
            edge_dir = "out" if id(data) in out_ids else "in"
            if direction != "both" and edge_dir != direction:
                continue
            if nbr not in self.touched and self._budget_left() <= 0:
                truncated = True
                break
            if len(edges) >= limit:
                truncated = True
                break
            self._touch(nbr)
            edges.append({"neighbor": self._stub(nn), "rel": self._rel_label(data),
                          "etype": data.get("etype"), "direction": edge_dir,
                          "confidence": round(float(data.get("confidence", 0.0)), 3),
                          "provenance": data.get("provenance")})
        return {"node": self._stub(node), "edges": edges, "truncated": truncated}

    # ---- find_path (bounded shortest path over the symmetrized projection) ---
    def _tool_find_path(self, source_id: str, target_id: str, max_hops=4) -> dict:
        try:
            max_hops = max(1, min(int(max_hops), self.config.agent_max_path_hops))
        except (TypeError, ValueError):
            max_hops = self.config.agent_max_path_hops
        if not self.store.get_node(source_id) or not self.store.get_node(target_id):
            return {"found": False, "reason": "source or target is not a known node"}
        if source_id == target_id:
            return {"found": False, "reason": "source and target are the same node"}
        G = self.projection()
        if source_id not in G or target_id not in G:
            return {"found": False, "reason": "a node is not in the traversal projection "
                    "(superseded or isolated)"}
        try:
            path = nx.shortest_path(G, source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {"found": False, "reason": "no path between the two nodes"}
        if len(path) - 1 > max_hops:
            return {"found": False, "reason": f"shortest path is {len(path) - 1} hops "
                    f"(> max_hops {max_hops})"}
        # annotate each hop with the real DIRECTED relationship label. The path nodes are
        # touched WHOLE (a deliberate bounded exception to the node budget): a partial chain
        # is meaningless, and the path is capped at agent_max_path_hops+1 (<=6 by default),
        # so the overrun is tiny. The endpoint-object harvest below DOES honour the budget.
        hop_rows, objects = [], []
        for i, nid in enumerate(path):
            n = self.store.get_node(nid)
            if n is None:
                continue
            self._touch(nid)
            row = self._stub(n)
            if i == 0:
                row["rel_in"], row["direction"] = None, None
            else:
                rel, edge_dir = self._directed_label(path[i - 1], nid)
                row["rel_in"], row["direction"] = rel, edge_dir
            hop_rows.append(row)
            if n.ntype == NodeType.OBJECT:
                objects.append(self._stub(n, snippet=True))
        # add a few object neighbors of the endpoints as citation candidates
        for endpoint in (source_id, target_id):
            for nbr, _d in self.store.neighbors(endpoint):
                nn = self.store.get_node(nbr)
                if nn and nn.ntype == NodeType.OBJECT and self._budget_left() > 0:
                    if nbr not in {o["id"] for o in objects}:
                        self._touch(nbr)
                        objects.append(self._stub(nn, snippet=True))
                if len(objects) >= 8:
                    break
        return {"found": True, "hops": len(path) - 1, "path": hop_rows, "objects": objects}

    def _directed_label(self, u: str, v: str) -> tuple[str, str]:
        """Real directed label/direction of the edge between consecutive path nodes."""
        for nbr, data in self.store.neighbors(u, direction="out"):
            if nbr == v:
                return self._rel_label(data), "out"
        for nbr, data in self.store.neighbors(u, direction="in"):
            if nbr == v:
                return self._rel_label(data), "in"
        return "related_to", "both"

    # ---- read_object (the only full-text tool) ------------------------------
    def _tool_read_object(self, object_id: str, max_chars=None) -> dict:
        node = self.store.get_node(object_id)
        if node is None:
            return {"error": f"no such node: {object_id}"}
        if node.ntype != NodeType.OBJECT:
            return {"error": f"node {object_id} is not an object (type={node.ntype.value}); "
                    "read_object only reads objects"}
        try:
            cap = int(max_chars) if max_chars else self.config.agent_read_chars
        except (TypeError, ValueError):
            cap = self.config.agent_read_chars
        cap = max(1, min(cap, self.config.agent_read_chars))
        self._touch(object_id)
        text = (node.raw_text or node.description or "")
        return {"id": node.id, "name": node.name, "modality": node.modality.value
                if node.modality else None, "source_ref": node.source_ref,
                "valid": node.valid, "tags": node.tags[:12], "text": text[:cap]}

    # ---- browse_themes (the only breadth tool) ------------------------------
    def _tool_browse_themes(self, query: str, k=5) -> dict:
        try:
            k = max(1, min(int(k), 8))
        except (TypeError, ValueError):
            k = 5
        hits = self.community.retrieve(query, k=k)
        if not hits:
            return {"themes": [], "note": "no communities built; run `python -m kg communities`"}
        themes, truncated = [], False
        for h in hits:
            # emit only the member ids that fit the node budget, so what the model SEES
            # equals what we counted (emitted == touched) — same invariant as the other tools.
            surfaced = []
            for m in h.get("members", [])[:8]:
                if m in self.touched or self._budget_left() > 0:
                    self._touch(m)
                    surfaced.append(m)
                else:
                    truncated = True
            themes.append({"community": h["community"], "score": h["score"],
                           "size": h["size"], "summary": (h.get("summary") or "")[:200],
                           "members": surfaced})
        return {"themes": themes, "truncated": truncated}


# --------------------------------------------------------------------------- #
# Shared citation validation + extractive synthesis (used by both backends)
# --------------------------------------------------------------------------- #
def _validate_citations(raw, trace: Trace, store: GraphStore) -> tuple[list[str], list[str]]:
    """A citation survives only if it is an OBJECT node the model actually READ. Everything
    else is dropped — the model structurally cannot cite what it never read."""
    kept, dropped, seen = [], [], set()
    for cid in (raw or []):
        if not isinstance(cid, str) or cid in seen:
            continue
        seen.add(cid)
        n = store.get_node(cid)
        if n is not None and n.ntype == NodeType.OBJECT and cid in trace.read:
            kept.append(cid)
        else:
            dropped.append(cid)
    return kept, dropped


def _ranked_object_ids(citations: list[str], touched: list[str],
                       store: GraphStore) -> list[str]:
    """Eval seam: citations first, then any other OBJECT the agent surfaced, deduped."""
    out, seen = [], set()
    for oid in [*citations, *touched]:
        if oid in seen:
            continue
        n = store.get_node(oid)
        if n is not None and n.ntype == NodeType.OBJECT:
            seen.add(oid)
            out.append(oid)
    return out


def _unwrap_submit_markup(text: str) -> tuple[str, list[str]]:
    """If a model wrote submit_answer as literal text instead of a tool_use block, return
    (inner answer prose, cited obj ids); otherwise return (text, []) unchanged."""
    if "submit_answer" not in text and 'name="answer"' not in text:
        return text, []
    m = _MARKUP_ANSWER.search(text)
    answer = m.group(1).strip() if m else text
    cm = _MARKUP_CITES.search(text)
    return answer, (_OBJ_ID.findall(cm.group(1)) if cm else [])


def _extractive_answer(query: str, read: list[Node], path_note: str | None = None) -> str:
    """Deterministic, grounded synthesis for the offline path: stitch the leads of the
    objects the agent read, plus an optional connection note."""
    if not read:
        return ("No supporting objects were found in the graph for this question."
                if not path_note else path_note)
    parts = [f"Based on {len(read)} object(s) in the graph:"]
    for n in read:
        parts.append(f"- {n.name or n.id}: {_snippet(n, 220)}")
    if path_note:
        parts.append(path_note)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# ClaudeAgent — the real Anthropic tool-use loop
# --------------------------------------------------------------------------- #
class ClaudeAgent:
    name = "claude"

    def __init__(self, tools: GraphTools, config: Config, *, client):
        self.tools = tools
        self.config = config
        self.client = client

    def run(self, query: str) -> AgentAnswer:
        if not query or not query.strip():
            return AgentAnswer(query=query, answer="(empty query)", backend=self.name)
        messages = [{"role": "user", "content": query}]
        all_tools = TOOLS + [SUBMIT_ANSWER_TOOL]
        trace = Trace()
        seen_calls: set[tuple] = set()
        steps = 0
        for step in range(self.config.agent_max_steps):
            try:
                msg = self.client.messages.create(
                    model=self.config.agent_model, max_tokens=self.config.agent_max_tokens,
                    temperature=0, system=_AGENT_SYS, tools=all_tools, messages=messages)
            except Exception as e:  # noqa: BLE001 — a transient API error must not crash
                # `kg ask`; degrade to a best-effort answer from evidence already gathered,
                # mirroring _force_submit / get_agent's degrade-don't-crash posture.
                return self._answer_from_evidence(query, trace, steps, "api_error",
                                                  extra_note=f"api error: {e!r}")
            messages.append({"role": "assistant", "content": msg.content})
            tool_uses = [b for b in msg.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:                       # model answered in prose, no tool call
                return self._finalize_from_text(query, msg, trace, steps)
            results = []
            for tu in tool_uses:
                if tu.name == "submit_answer":
                    return self._build_answer(query, tu.input, trace, steps + 1, "answered")
                key = (tu.name, json.dumps(tu.input, sort_keys=True))
                if key in seen_calls:
                    out = {"note": "already called with these arguments; use the prior "
                           "result or answer now"}
                else:
                    out = self.tools.dispatch(tu.name, tu.input)
                    seen_calls.add(key)
                trace.record(step, tu.name, tu.input, out)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": json.dumps(out)[:self.config.agent_result_chars]})
            messages.append({"role": "user", "content": results})
            steps = step + 1
        return self._force_submit(query, messages, trace, steps)

    # ---- terminal builders --------------------------------------------------
    def _build_answer(self, query, submit_input, trace, steps, stopped) -> AgentAnswer:
        submit_input = submit_input or {}
        answer = str(submit_input.get("answer", "")).strip()
        kept, dropped = _validate_citations(submit_input.get("citations", []), trace,
                                            self.tools.store)
        return self._assemble(query, answer, kept, dropped, trace, steps, stopped)

    def _finalize_from_text(self, query, msg, trace, steps) -> AgentAnswer:
        """Tolerant path: the model answered in prose without submit_answer. Salvage the
        text and any obj_* ids it mentioned, then validate them like real citations."""
        text = " ".join(b.text for b in msg.content
                        if getattr(b, "type", None) == "text").strip()
        answer, raw = _unwrap_submit_markup(text)   # strip literal tool-call markup if present
        if not raw:
            raw = _OBJ_ID.findall(text)
        kept, dropped = _validate_citations(raw, trace, self.tools.store)
        return self._assemble(query, answer or "(no answer produced)", kept, dropped,
                              trace, steps, "prose")

    def _force_submit(self, query, messages, trace, steps) -> AgentAnswer:
        """Budget hit — force one structured submit_answer turn from evidence gathered.

        The loop always ends on a `user` (tool_results) turn, so we fold the budget nudge
        INTO that turn rather than appending a second consecutive user message (which the
        Anthropic API rejects — roles must alternate). tool_choice forces submit_answer; the
        full tool set is passed so history tool_use blocks stay defined."""
        nudge = ("You have reached the exploration budget. Call submit_answer now with your "
                 "best answer from the evidence gathered so far.")
        msgs = list(messages)
        if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
            msgs[-1] = {"role": "user",
                        "content": msgs[-1]["content"] + [{"type": "text", "text": nudge}]}
        else:
            msgs = msgs + [{"role": "user", "content": nudge}]
        try:
            msg = self.client.messages.create(
                model=self.config.agent_model, max_tokens=self.config.agent_max_tokens,
                temperature=0, system=_AGENT_SYS, tools=TOOLS + [SUBMIT_ANSWER_TOOL],
                tool_choice={"type": "tool", "name": "submit_answer"}, messages=msgs)
            for b in msg.content:
                if getattr(b, "type", None) == "tool_use" and b.name == "submit_answer":
                    return self._build_answer(query, b.input, trace, steps, "step_cap")
        except Exception:  # noqa: BLE001 — fall through to an evidence-only answer
            pass
        return self._answer_from_evidence(query, trace, steps, "step_cap")

    def _answer_from_evidence(self, query, trace, steps, stopped, extra_note=None) -> AgentAnswer:
        """Best-effort extractive answer from the objects already read — so a run that hits
        the step cap or a mid-loop API error still returns a grounded AgentAnswer, never
        nothing. Citations are the read objects (they pass the same read-gate)."""
        read_nodes = [n for n in (self.tools.store.get_node(r) for r in trace.read) if n]
        ans = self._assemble(query, _extractive_answer(query, read_nodes),
                             [n.id for n in read_nodes], [], trace, steps, stopped)
        if extra_note:
            ans.notes.append(extra_note)
        return ans

    def _assemble(self, query, answer, kept, dropped, trace, steps, stopped) -> AgentAnswer:
        touched = self.tools.touched_ids()
        notes = []
        if dropped:
            notes.append(f"dropped {len(dropped)} unvalidated citation(s): {', '.join(dropped)}")
        return AgentAnswer(
            query=query, answer=answer, citations=kept, dropped_citations=dropped,
            backend=self.name, steps=steps, stopped=stopped, trace=trace.steps,
            seeds=trace.seeds, touched=touched,
            object_ids=_ranked_object_ids(kept, touched, self.tools.store), notes=notes)


# --------------------------------------------------------------------------- #
# OfflineAgent — deterministic, no-key policy over the SAME executors
# --------------------------------------------------------------------------- #
class OfflineAgent:
    name = "offline"

    def __init__(self, tools: GraphTools, config: Config):
        self.tools = tools
        self.config = config

    def _call(self, trace: Trace, step: int, name: str, inp: dict) -> dict:
        out = self.tools.dispatch(name, inp)
        trace.record(step, name, inp, out)
        return out

    def run(self, query: str) -> AgentAnswer:
        if not query or not query.strip():
            return AgentAnswer(query=query, answer="(empty query)", backend=self.name)
        trace = Trace()
        store = self.tools.store

        # 1. global breadth → themes
        if (is_global_query(query)
                and self.tools.store.nodes_of_type(NodeType.COMMUNITY)):
            out = self._call(trace, 0, "browse_themes", {"query": query, "k": 5})
            themes = out.get("themes", [])
            if themes:
                cand: list[str] = []
                for t in themes[:3]:
                    cand.extend(t.get("members", [])[:3])
                read = self._read_candidates(trace, cand, start_step=1)
                summary = " ".join(t["summary"] for t in themes[:3])
                answer = f"Main themes for {query!r}: {summary}".strip()
                return self._assemble(query, answer, [n.id for n in read], trace, "answered")

        # 2. local entry: seed-and-spread (PPR)
        step = 0
        out = self._call(trace, step, "seed_and_spread", {"query": query, "k": self.config.top_k})
        objects = out.get("objects", [])
        seeds = out.get("seeds", [])

        # 3. escalation ladder when PPR is thin
        top_score = objects[0]["score"] if objects else 0.0
        if not objects or top_score < self.config.agent_offline_floor:
            step += 1
            kw = self._call(trace, step, "keyword_search", {"query": query, "k": self.config.top_k})
            step += 1
            ve = self._call(trace, step, "vector_search", {"query": query, "k": self.config.top_k})
            objects = _merge_object_rows(objects, kw.get("objects", []), ve.get("objects", []))

        candidates = [o["id"] for o in objects]
        path_note = None
        nb_edges: list[dict] = []

        # 4a. one-hop enrich: a real edge-semantics row from the top object
        if candidates:
            step += 1
            nb = self._call(trace, step, "neighbors",
                            {"node_id": candidates[0], "direction": "both", "limit": 6})
            nb_edges = nb.get("edges", [])
            for e in nb_edges:
                nbr = e["neighbor"]
                if nbr["ntype"] == "object" and nbr["id"] not in candidates:
                    candidates.append(nbr["id"])
                    if e.get("confidence", 0) >= 0.5:
                        break

        # 4b. connection question → find_path over the two highest-IDF entities. Entity
        # ids come from the seeds and the entities the top object MENTIONS; if those don't
        # yield two, a MENTIONS-scoped neighbors call harvests them, so a connection
        # question reliably reaches find_path even when seeds were object/tag-dominated.
        if _CONNECT_CUE.search(query) and candidates:
            ent_ids = [s["id"] for s in seeds if s["ntype"] == "entity"]
            ent_ids += [e["neighbor"]["id"] for e in nb_edges
                        if e["neighbor"]["ntype"] == "entity"]
            if len(set(ent_ids)) < 2:
                step += 1
                ment = self._call(trace, step, "neighbors",
                                  {"node_id": candidates[0], "etypes": ["MENTIONS"],
                                   "direction": "out", "limit": 8})
                ent_ids += [e["neighbor"]["id"] for e in ment.get("edges", [])
                            if e["neighbor"]["ntype"] == "entity"]
            ent_ids = list(dict.fromkeys(ent_ids))
        else:
            ent_ids = []
        if len(ent_ids) >= 2:
            ent_ids.sort(key=lambda e: self.tools.canon.idf_weight(e), reverse=True)
            step += 1
            fp = self._call(trace, step, "find_path",
                            {"source_id": ent_ids[0], "target_id": ent_ids[1],
                             "max_hops": self.config.agent_max_path_hops})
            if fp.get("found"):
                chain = " -> ".join(f"{h['name']}"
                                    + (f" [{h['rel_in']}]" if h.get("rel_in") else "")
                                    for h in fp.get("path", []))
                path_note = f"Connection: {chain}."
                for o in fp.get("objects", []):
                    if o["id"] not in candidates:
                        candidates.append(o["id"])

        # 5. read the top candidates and synthesize an extractive answer
        read = self._read_candidates(trace, candidates[:max(3, self.config.top_k)],
                                     start_step=step + 1, limit=min(3, self.config.top_k))
        answer = _extractive_answer(query, read, path_note)
        return self._assemble(query, answer, [n.id for n in read], trace, "answered")

    def _read_candidates(self, trace, candidate_ids, start_step, limit=3) -> list[Node]:
        read = []
        s = start_step
        for cid in candidate_ids:
            if len(read) >= limit:
                break
            out = self._call(trace, s, "read_object", {"object_id": cid, "max_chars": 600})
            s += 1
            if not out.get("error"):
                n = self.tools.store.get_node(cid)
                if n:
                    read.append(n)
        return read

    def _assemble(self, query, answer, kept, trace, stopped) -> AgentAnswer:
        kept, dropped = _validate_citations(kept, trace, self.tools.store)
        touched = self.tools.touched_ids()
        return AgentAnswer(
            query=query, answer=answer, citations=kept, dropped_citations=dropped,
            backend=self.name, steps=len(trace.steps), stopped=stopped, trace=trace.steps,
            seeds=trace.seeds, touched=touched,
            object_ids=_ranked_object_ids(kept, touched, self.tools.store),
            notes=["degraded to offline agent"] if self.config.agent_backend == "auto" else [])


def _merge_object_rows(*rowsets) -> list[dict]:
    """Union object-stub rows from several search tools, keeping the best score per id."""
    by_id: dict[str, dict] = {}
    for rows in rowsets:
        for r in rows:
            cur = by_id.get(r["id"])
            if cur is None or r.get("score", 0) > cur.get("score", 0):
                by_id[r["id"]] = r
    return sorted(by_id.values(), key=lambda r: -r.get("score", 0))


# --------------------------------------------------------------------------- #
# Factory (mirrors get_extractor / get_embedder)
# --------------------------------------------------------------------------- #
def get_agent(store: GraphStore, embedder: Embedder, canon: Canonicalizer,
              config: Config, *, client=None):
    """Real Anthropic agent when a client/key is available, else the deterministic offline
    agent — so `python -m kg ask` and every test run fully offline by default. `client=`
    lets tests inject a scripted fake."""
    tools = GraphTools(store, embedder, canon, config)
    if client is not None:                              # injected (tests) or pre-built
        return ClaudeAgent(tools, config, client=client)
    if config.agent_backend == "offline":
        return OfflineAgent(tools, config)
    if config.agent_backend in ("auto", "claude") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            return ClaudeAgent(tools, config, client=anthropic.Anthropic())
        except Exception:  # noqa: BLE001 — missing dep / bad env → offline parity
            pass
    return OfflineAgent(tools, config)
