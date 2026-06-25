"""Graph-RAG answer flow (docs/ARCHITECTURE.md §5) — retrieve-then-read.

This is the query path the design favours: **the LLM does NOT traverse the graph.** A
non-LLM retriever (Personalized PageRank over the symmetrized, temporally-filtered
projection) does the multi-hop work and assembles a compact context — the top episodes'
text plus the currently-valid facts among the touched entities — and then a SINGLE LLM
call answers over that context with citations. No per-hop tool loop, no LLM-in-the-walk.

It mirrors the extractor's real⇄offline contract: a `ClaudeAnswerer` (one Anthropic call)
that degrades to a deterministic `OfflineAnswerer` (extractive synthesis over the same
context), so `kg ask` and every test run fully offline with no API key. `client=` injects a
(possibly fake) Anthropic client for tests.

Point-in-time: pass `as_of=T` to answer "as of T" — retrieval keeps only facts whose valid
window contained T, so "where did Becky live in 2022?" reads the world as it was then.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .canonicalize import Canonicalizer
from .config import Config
from .embedders import Embedder
from .metering import UsageMeter, empty_totals
from .models import EdgeType, NodeType
from .retrieval import PPRRetriever, RetrievalResult
from .store import GraphStore, fact_active

_WS = re.compile(r"\s+")
_EP_ID = re.compile(r"\bep_[A-Za-z0-9_#]+\b")


@dataclass
class FactLine:
    src: str
    rel: str
    dst: str
    valid_at: str = ""
    invalid_at: str = ""
    episode_id: str = ""

    def render(self) -> str:
        win = []
        if self.valid_at:
            win.append(f"since {self.valid_at[:10]}")
        if self.invalid_at:
            win.append(f"until {self.invalid_at[:10]}")
        w = f" ({'; '.join(win)})" if win else ""
        prov = f" [{self.episode_id}]" if self.episode_id else ""
        return f"{self.src} --{self.rel}--> {self.dst}{w}{prov}"


@dataclass
class RagAnswer:
    query: str
    answer: str
    citations: list[str] = field(default_factory=list)        # episode ids used
    dropped_citations: list[str] = field(default_factory=list)
    backend: str = "offline"
    mode: str = "rag"
    as_of: str | None = None
    context_episodes: list[str] = field(default_factory=list)  # episode ids in the context
    facts: list[str] = field(default_factory=list)             # rendered fact lines
    object_ids: list[str] = field(default_factory=list)        # PPR ranking (eval seam)
    seeds: list[str] = field(default_factory=list)
    touched: list[str] = field(default_factory=list)           # every node in the PPR subgraph
    usage: dict = field(default_factory=dict)                  # token/cost (empty offline)
    steps: int = 1            # retrieve-then-read = ONE answer call (no per-hop loop)
    stopped: str = "answered"
    trace: list = field(default_factory=list)                  # no tool trace (RAG, not agentic)
    notes: list[str] = field(default_factory=list)


_RAG_SYS = (
    "You answer a question using ONLY the EPISODES and FACTS provided in the context — a "
    "knowledge graph already retrieved the relevant evidence for you. Do not use outside "
    "knowledge and do not invent facts. The FACTS section lists relationships that are "
    "currently valid (or valid at the requested point in time); a relationship NOT listed "
    "is not currently true even if an episode once stated it (the graph tracks when facts "
    "end). Prefer the FACTS for state questions (who/where/what is X now). Cite the episode "
    "ids (e.g. ep_3) you relied on. If the context does not answer the question, say so."
)

_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the final answer grounded in the provided context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"},
                          "description": "episode ids you used, e.g. ep_2"},
        },
        "required": ["answer", "citations"],
    },
}


# --------------------------------------------------------------------------- #
# Context builder (shared by both backends) — this is where PPR's subgraph becomes RAG
# --------------------------------------------------------------------------- #
class ContextBuilder:
    def __init__(self, store: GraphStore, config: Config):
        self.store = store
        self.config = config

    def _snippet(self, node, n: int) -> str:
        text = node.raw_text or node.description or node.summary or node.name or ""
        return _WS.sub(" ", text).strip()[:n]

    def relevant_entities(self, result: RetrievalResult, episodes: list[str]) -> list[str]:
        """Entities that anchor the answer: those in the PPR subgraph, plus the entities
        the top episodes mention (via the mention star)."""
        ents: list[str] = []
        seen: set[str] = set()

        def add(eid):
            if eid not in seen and self.store.get_node(eid) and \
                    self.store.get_node(eid).ntype == NodeType.ENTITY:
                seen.add(eid)
                ents.append(eid)

        for nid in result.subgraph:
            add(nid)
        for ep in episodes:
            for mid, _d in self.store.neighbors(ep, etypes={EdgeType.MENTIONED_IN},
                                                direction="in"):
                for eid, _d2 in self.store.neighbors(mid, etypes={EdgeType.RESOLVES_TO},
                                                     direction="out"):
                    add(eid)
        return ents

    def facts_for(self, entities: list[str], as_of: str | None) -> list[FactLine]:
        """Currently-valid (or as-of-T) facts touching the relevant entities. Walks BOTH
        directions (a symmetric fact is stored in one orientation, so an anchor entity may
        be the edge's destination) and dedupes, so no valid fact is dropped or double-listed."""
        out: list[FactLine] = []
        seen: set[tuple] = set()
        for eid in entities:
            for direction in ("out", "in"):
                for nbr, data in self.store.neighbors(eid, etypes={EdgeType.RELATED_TO},
                                                      direction=direction):
                    if not fact_active(data, as_of):
                        continue
                    src_id, dst_id = (eid, nbr) if direction == "out" else (nbr, eid)
                    fkey = (src_id, data.get("rel_tag"), dst_id, data.get("valid_at", ""))
                    if fkey in seen:
                        continue
                    seen.add(fkey)
                    rel_node = self.store.get_node(data.get("rel_tag")) if data.get("rel_tag") else None
                    sn, tn = self.store.get_node(src_id), self.store.get_node(dst_id)
                    out.append(FactLine(
                        src=sn.name if sn else src_id,
                        rel=rel_node.name if rel_node else "related_to",
                        dst=tn.name if tn else dst_id, valid_at=data.get("valid_at", ""),
                        invalid_at=data.get("invalid_at", ""),
                        episode_id=data.get("episode_id", "")))
                    if len(out) >= self.config.rag_max_facts:
                        return out
        return out

    def build(self, result: RetrievalResult) -> tuple[list[str], list[FactLine], str]:
        """Return (episode_ids, fact_lines, context_blob)."""
        ep_ids = result.object_ids[: self.config.rag_context_episodes]
        ents = self.relevant_entities(result, ep_ids)
        facts = self.facts_for(ents, result.as_of)

        lines = [f"QUESTION: {result.query}",
                 f"AS-OF: {result.as_of or 'now (current view)'}", ""]
        lines.append("EPISODES (evidence; cite by id):")
        if ep_ids:
            for eid in ep_ids:
                n = self.store.get_node(eid)
                if not n:
                    continue
                when = (n.created_at or "")[:10]
                lines.append(f"[{eid}] ({when}) {n.name}: "
                             f"{self._snippet(n, self.config.rag_episode_chars)}")
        else:
            lines.append("(none retrieved)")
        lines.append("")
        lines.append("FACTS currently valid among the relevant entities:")
        lines += [f"- {f.render()}" for f in facts] or ["(none)"]
        return ep_ids, facts, "\n".join(lines)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _validate(raw, context_ids: list[str]) -> tuple[list[str], list[str]]:
    """A citation survives only if it names an episode that was actually in the context."""
    allow, kept, dropped, seen = set(context_ids), [], [], set()
    for cid in (raw or []):
        if not isinstance(cid, str) or cid in seen:
            continue
        seen.add(cid)
        (kept if cid in allow else dropped).append(cid)
    return kept, dropped


def _extractive(store: GraphStore, query: str, ep_ids: list[str],
                facts: list[FactLine]) -> str:
    """Deterministic, grounded synthesis for the offline path: the relevant facts first
    (they answer state questions directly), then the supporting episode leads."""
    if not ep_ids and not facts:
        return "No supporting episodes or facts were found in the graph for this question."
    parts = []
    if facts:
        parts.append("Relevant current facts:")
        parts += [f"- {f.render()}" for f in facts]
    if ep_ids:
        parts.append(f"Supported by {len(ep_ids)} episode(s):")
        for eid in ep_ids:
            n = store.get_node(eid)
            if n:
                txt = _WS.sub(" ", (n.raw_text or n.description or "")).strip()[:220]
                parts.append(f"- [{eid}] {n.name}: {txt}")
    return "\n".join(parts)


class ClaudeAnswerer:
    name = "claude"

    def __init__(self, store, config: Config, builder: ContextBuilder, *, client):
        self.store = store
        self.config = config
        self.builder = builder
        self.client = client
        self.meter = UsageMeter()

    def answer(self, result: RetrievalResult) -> RagAnswer:
        ep_ids, facts, blob = self.builder.build(result)
        base = RagAnswer(query=result.query, answer="", backend=self.name,
                         as_of=result.as_of, context_episodes=ep_ids,
                         facts=[f.render() for f in facts], object_ids=result.object_ids,
                         seeds=result.seeds, touched=sorted(result.subgraph))
        try:
            msg = self.client.messages.create(
                model=self.config.rag_model, max_tokens=self.config.rag_max_tokens,
                temperature=0, system=_RAG_SYS, tools=[_ANSWER_TOOL],
                tool_choice={"type": "tool", "name": "submit_answer"},
                messages=[{"role": "user", "content": blob}])
            self.meter.record("rag", self.config.rag_model, msg, label=result.query[:40])
        except Exception as e:  # noqa: BLE001 — degrade to the offline synthesis, never crash
            base.answer = _extractive(self.store, result.query, ep_ids, facts)
            base.citations = ep_ids
            base.usage = self.meter.totals()
            base.notes.append(f"api error, used extractive fallback: {e!r}")
            return base
        base.usage = self.meter.totals()
        ans, raw = "", []
        for b in msg.content:
            if getattr(b, "type", None) == "tool_use" and b.name == "submit_answer":
                ans = str(b.input.get("answer", "")).strip()
                raw = b.input.get("citations", [])
            elif getattr(b, "type", None) == "text":
                ans = ans or b.text.strip()
        if not raw:
            raw = _EP_ID.findall(ans)
        kept, dropped = _validate(raw, ep_ids)
        base.answer = ans or "(no answer produced)"
        base.citations, base.dropped_citations = kept, dropped
        if dropped:
            base.notes.append(f"dropped {len(dropped)} uncontextual citation(s)")
        return base


class OfflineAnswerer:
    name = "offline"

    def __init__(self, store, config: Config, builder: ContextBuilder):
        self.store = store
        self.config = config
        self.builder = builder

    def answer(self, result: RetrievalResult) -> RagAnswer:
        ep_ids, facts, _blob = self.builder.build(result)
        return RagAnswer(
            query=result.query, answer=_extractive(self.store, result.query, ep_ids, facts),
            citations=ep_ids, backend=self.name, as_of=result.as_of,
            context_episodes=ep_ids, facts=[f.render() for f in facts],
            object_ids=result.object_ids, seeds=result.seeds,
            touched=sorted(result.subgraph), usage=empty_totals(),
            notes=["degraded to offline answerer"] if self.config.rag_backend == "auto" else [])


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class RagAnswerer:
    """PPR-retrieve → build context → single answer call. The public `ask` entry point."""

    def __init__(self, store: GraphStore, embedder: Embedder, canon: Canonicalizer,
                 config: Config, *, client=None):
        self.store = store
        self.config = config
        self.retriever = PPRRetriever(store, embedder, canon, config)
        self.builder = ContextBuilder(store, config)
        self._backend = self._pick_backend(client)

    def _pick_backend(self, client):
        if client is not None:
            return ClaudeAnswerer(self.store, self.config, self.builder, client=client)
        if self.config.rag_backend == "offline":
            return OfflineAnswerer(self.store, self.config, self.builder)
        if self.config.rag_backend in ("auto", "claude") and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                return ClaudeAnswerer(self.store, self.config, self.builder,
                                      client=anthropic.Anthropic())
            except Exception:  # noqa: BLE001 — missing dep / bad env → offline parity
                pass
        return OfflineAnswerer(self.store, self.config, self.builder)

    def run(self, query: str, k: int | None = None, as_of: str | None = None) -> RagAnswer:
        if not query or not query.strip():
            return RagAnswer(query=query, answer="(empty query)",
                             backend=self._backend.name, as_of=as_of)
        result = self.retriever.retrieve(query, k=k or self.config.top_k, as_of=as_of)
        return self._backend.answer(result)


def get_answerer(store, embedder, canon, config, *, client=None) -> RagAnswerer:
    return RagAnswerer(store, embedder, canon, config, client=client)
