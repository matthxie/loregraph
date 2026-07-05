"""Graph-RAG answer flow (docs/ARCHITECTURE.md §5) — retrieve-then-read.

This is the query path the design favours: **the LLM does NOT traverse the graph.** A
non-LLM retriever (Personalized PageRank over the symmetrized, temporally-filtered
projection) does the multi-hop work and assembles a compact context — the top episodes'
text plus the currently-valid facts among the touched entities — and then a SINGLE LLM
call answers over that context with citations. No per-hop tool loop, no LLM-in-the-walk.

Answering is live-only: an `OpenAIAnswerer` makes one OpenAI call and validates citations.
`client=` injects a (possibly fake) OpenAI client for tests. The selectable offline
answerer was removed; a deterministic extractive synthesis (`_extractive`) survives ONLY as
an internal crash-guard if that single live call raises mid-run, so one transient API error
never sinks a whole test run — it is not a user-facing backend.

Point-in-time: pass `as_of=T` to answer "as of T" — retrieval keeps only facts whose valid
window contained T, so "where did Becky live in 2022?" reads the world as it was then.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .backoff import call_with_backoff
from .canonicalize import Canonicalizer
from .config import Config
from .embedders import Embedder
from .facts import FactIndex, FactLine
from .metering import UsageMeter
from .models import EdgeType, NodeType
from .profiler import span as prof_span
from .retrieval import HybridRetriever, RetrievalResult
from .route import STATE
from .store import GraphStore, fact_active

_WS = re.compile(r"\s+")
_EP_ID = re.compile(r"\bep_[A-Za-z0-9_#]+\b")
_CHUNK_ID = re.compile(r"^(.+)#c(\d+)$")



@dataclass
class RagAnswer:
    query: str
    answer: str
    citations: list[str] = field(default_factory=list)        # episode ids used
    dropped_citations: list[str] = field(default_factory=list)
    backend: str = "openai"
    mode: str = "rag"
    as_of: str | None = None
    context_episodes: list[str] = field(default_factory=list)  # episode ids in the context
    facts: list[str] = field(default_factory=list)             # rendered fact lines
    object_ids: list[str] = field(default_factory=list)        # PPR ranking (eval seam)
    ppr_pool: list = field(default_factory=list)               # (ep_id, raw PPR score) pool
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
    "ids (e.g. ep_3) you relied on. If the context does not answer the question, say so.\n"
    "For questions asking how many, how long, or how many days between: first list every "
    "matching event from the context with its date, then derive the number from that list — "
    "never state a count without enumerating the events behind it. For date arithmetic, "
    "compute the day difference step by step (months and remaining days) before answering.\n"
    "Verify the exact subject of the question appears in the context. If the context only "
    "contains a similar but different item (e.g. asked about vintage films but the context "
    "only has vintage cameras), say you don't have that information instead of substituting "
    "the similar item. This applies especially to place names and dates: if asked about "
    "city/venue X and the context only covers a different city/venue Y, say the information "
    "is not available — do not answer about Y.\n"
    "The FACTS lines are machine-extracted and may be wrong or mis-dated; the EPISODES text "
    "is the ground truth — never state a date or place that appears only in a FACTS line "
    "without confirming it in an episode.\n"
    "When the same running total is restated at different dates ('I've added N since X', "
    "'I now have N'), the most recent statement supersedes earlier ones — report the latest "
    "total; never add restatements together. Only sum amounts that are explicitly separate "
    "events."
)

_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit the final answer grounded in the provided context.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"},
                              "description": "episode ids you used, e.g. ep_2"},
            },
            "required": ["answer", "citations"],
        },
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

    def _select_episodes(self, ranked: list[str]) -> list[str]:
        """Top-n context episodes, with at most `rag_chunks_per_source` chunks of any
        one source (chunk ids are `ep_<source>#cNNN`) — otherwise a single chunked
        session can monopolize every slot and crowd out the other evidence."""
        n = self.config.rag_context_episodes
        cap = int(getattr(self.config, "rag_chunks_per_source", 0))
        if not cap:
            return ranked[:n]
        out: list[str] = []
        per: dict[str, int] = {}
        for eid in ranked:
            base = eid.split("#", 1)[0]
            if per.get(base, 0) >= cap:
                continue
            per[base] = per.get(base, 0) + 1
            out.append(eid)
            if len(out) >= n:
                break
        return out

    def _expand_siblings(self, selected: list[str]) -> list[str]:
        """Sibling-chunk expansion (rag_parent_expand): pull in each selected chunk's
        #cNNN neighbours within the configured radius, so a chunked session's
        answer-bearing sibling isn't dropped from context just because it didn't rank
        into the top-n itself. Context-only — the caller's retrieval-metric bookkeeping
        keys off the pre-expansion ranked list, not this. Inserted in document order
        (contiguous by chunk index within a source) and capped by
        rag_expand_budget_chars; sources are expanded in their original (rank) order, so
        when the budget runs out it is the lowest-ranked source's siblings that get cut,
        never an originally selected chunk."""
        w = int(getattr(self.config, "rag_parent_expand", 0) or 0)
        if w <= 0:
            return selected

        selected_set = set(selected)
        groups_order: list[str] = []          # group key, in first-appearance (rank) order
        base_idxs: dict[str, set[int]] = {}   # chunked group -> selected chunk indices
        for eid in selected:
            m = _CHUNK_ID.match(eid)
            key = m.group(1) if m else eid
            if key not in base_idxs and key not in groups_order:
                groups_order.append(key)
            if m:
                base_idxs.setdefault(key, set()).add(int(m.group(2)))

        def text_len(eid: str) -> int:
            n = self.store.get_node(eid)
            return len(self._snippet(n, self.config.rag_episode_chars)) if n else 0

        used_chars = sum(text_len(eid) for eid in selected)
        extra: dict[str, set[int]] = {base: set() for base in base_idxs}
        budget = int(getattr(self.config, "rag_expand_budget_chars", 60000))
        budget_hit = False
        for base in groups_order:                 # best-ranked source expanded first
            if budget_hit or base not in base_idxs:
                continue
            for idx in sorted(base_idxs[base]):
                if budget_hit:
                    break
                for cand in range(idx - w, idx + w + 1):
                    if cand < 0 or cand in base_idxs[base] or cand in extra[base]:
                        continue
                    cid = f"{base}#c{cand:03d}"
                    if cid in selected_set:
                        continue
                    length = text_len(cid)
                    if length == 0:               # sibling doesn't exist in the store
                        continue
                    if used_chars + length > budget:
                        budget_hit = True
                        break
                    extra[base].add(cand)
                    used_chars += length

        out: list[str] = []
        for key in groups_order:
            if key in base_idxs:
                idxs = sorted(base_idxs[key] | extra.get(key, set()))
                out.extend(f"{key}#c{i:03d}" for i in idxs)
            else:
                out.append(key)
        return out

    def build(self, result: RetrievalResult) -> tuple[list[str], list[FactLine], str]:
        """Return (episode_ids, fact_lines, context_blob)."""
        ep_ids = self._select_episodes(result.object_ids)
        ents = self.relevant_entities(result, ep_ids)
        facts = self.facts_for(ents, result.as_of)
        ctx_ids = self._expand_siblings(ep_ids)

        lines = [f"QUESTION: {result.query}",
                 f"AS-OF: {result.as_of or 'now (current view)'}", ""]
        lines.append("EPISODES (evidence; cite by id):")
        if ctx_ids:
            for eid in ctx_ids:
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

        # STATE/evolution lane: append the FULL closed+open fact history so "how has X
        # changed" can read the trajectory (the currently-valid FACTS above show only the
        # open state). Only fires when the router tagged this a STATE question AND there is
        # ended history — so plain `query`-mode results (no lane) are unaffected.
        if getattr(result, "lane", "single") == STATE:
            ent_ids = getattr(result, "entity_ids", []) or ents
            hist = FactIndex(self.store).history(ent_ids)
            if any(h.invalid_at for h in hist):
                lines += ["", "HISTORY (includes ENDED facts; read the trajectory in time order):"]
                lines += [f"- {h.render()}" for h in hist]
        return ctx_ids, facts, "\n".join(lines)


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


class OpenAIAnswerer:
    name = "openai"

    def __init__(self, store, config: Config, builder: ContextBuilder, *, client):
        self.store = store
        self.config = config
        self.builder = builder
        self.client = client
        self.meter = UsageMeter()

    def answer(self, result: RetrievalResult) -> RagAnswer:
        with prof_span("query.build_context"):
            ep_ids, facts, blob = self.builder.build(result)
        base = RagAnswer(query=result.query, answer="", backend=self.name,
                         as_of=result.as_of, context_episodes=ep_ids,
                         facts=[f.render() for f in facts], object_ids=result.object_ids,
                         seeds=result.seeds, touched=sorted(result.subgraph))
        # gpt-5 / o-series models reject `max_tokens` (they want max_completion_tokens)
        # and any non-default temperature; 4o-era models keep the old params. Getting this
        # wrong would not crash loudly — the except below silently degrades every answer to
        # the offline extractive path — so the split must live here.
        kwargs: dict = {
            "model": self.config.rag_model,
            "messages": [
                {"role": "system", "content": _RAG_SYS},
                {"role": "user", "content": blob},
            ],
            "tools": [_ANSWER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "submit_answer"}},
        }
        if self.config.rag_model.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["max_completion_tokens"] = self.config.rag_max_tokens
        else:
            kwargs["max_tokens"] = self.config.rag_max_tokens
            kwargs["temperature"] = 0
        try:
            with prof_span("query.llm_answer"):
                msg = call_with_backoff(lambda: self.client.chat.completions.create(**kwargs))
            self.meter.record("rag", self.config.rag_model, msg, label=result.query[:40])
        except Exception as e:  # noqa: BLE001 — degrade to the offline synthesis, never crash
            base.answer = _extractive(self.store, result.query, ep_ids, facts)
            base.citations = ep_ids
            base.usage = self.meter.totals()
            base.notes.append(f"api error, used extractive fallback: {e!r}")
            return base
        base.usage = self.meter.totals()
        ans, raw = "", []
        tc = getattr(msg.choices[0].message, "tool_calls", None) if msg.choices else None
        if tc and tc[0].function.name == "submit_answer":
            payload = json.loads(tc[0].function.arguments)
            ans = str(payload.get("answer", "")).strip()
            raw = payload.get("citations", [])
        elif msg.choices and msg.choices[0].message.content:
            ans = msg.choices[0].message.content.strip()
        if not raw:
            raw = _EP_ID.findall(ans)
        kept, dropped = _validate(raw, ep_ids)
        base.answer = ans or "(no answer produced)"
        base.citations, base.dropped_citations = kept, dropped
        if dropped:
            base.notes.append(f"dropped {len(dropped)} uncontextual citation(s)")
        return base


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class RagAnswerer:
    """Hybrid-retrieve → build context → single answer call. The public `ask` entry point.
    The retriever routes the question, augments state/evolution lanes with fact-bearing
    episodes, and reranks the hard lanes with a cross-encoder; the LLM never traverses."""

    def __init__(self, store: GraphStore, embedder: Embedder, canon: Canonicalizer,
                 config: Config, *, client=None):
        self.store = store
        self.config = config
        self.retriever = HybridRetriever(store, embedder, canon, config)
        self.builder = ContextBuilder(store, config)
        self._backend = self._pick_backend(client)

    def _pick_backend(self, client):
        """Live-only: an OpenAIAnswerer over an injected client, else a real OpenAI client
        from the env key. There is no offline backend — without a key (and no injected
        client) we raise, rather than silently degrade to a fake answer."""
        if client is not None:
            return OpenAIAnswerer(self.store, self.config, self.builder, client=client)
        if os.environ.get("OPENAI_API_KEY"):
            import openai
            return OpenAIAnswerer(self.store, self.config, self.builder,
                                  client=openai.OpenAI())
        raise RuntimeError(
            "No OPENAI_API_KEY found. The query/answer path is live-only. "
            "Set the key (kg auto-reads a project-root .env), or "
            "inject a client: get_answerer(..., client=fake).")

    def run(self, query: str, k: int | None = None, as_of: str | None = None,
            kind: str | None = None) -> RagAnswer:
        if not query or not query.strip():
            return RagAnswer(query=query, answer="(empty query)",
                             backend=self._backend.name, as_of=as_of)
        result = self.retriever.retrieve(query, k=k or self.config.top_k, as_of=as_of,
                                         kind=kind)
        ans = self._backend.answer(result)
        ans.ppr_pool = list(getattr(result, "ppr_pool", []) or [])
        ans.lane = getattr(result, "lane", "")            # surface the routed lane
        ans.rerank_active = self.retriever.rerank_active
        return ans


def get_answerer(store, embedder, canon, config, *, client=None) -> RagAnswerer:
    return RagAnswerer(store, embedder, canon, config, client=client)
