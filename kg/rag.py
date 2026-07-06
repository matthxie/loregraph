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
_WORD = re.compile(r"[a-z0-9]+")
# Payload signals for the lexical retarget scorer (rag_retarget="seed+lex"): a chunk that
# actually carries a date/number/amount is preferred over one that merely echoes the
# question's wording with none — the failure this guards is a chunk like "I'm preparing
# for an upcoming meeting..." (high word overlap with the question, zero payload) beating
# the chunks that carry the actual dates being asked about. Deliberately simple/generic
# (reused verbatim in spirit from kg/cues.py's quantity screen, not imported — that
# module's regexes are tuned for the ingest-time cue-gating decision, a different job).
_PAYLOAD_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE)
_PAYLOAD_CURRENCY = re.compile(
    r"[$€£¥₹]\s?\d[\d,]*(?:\.\d+)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:dollars?|bucks|cents?|usd|euros?|quid)\b",
    re.IGNORECASE)
_PAYLOAD_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
# Relative-date phrases carry no digits ("a month ago", "last week", "exactly two months
# ago") so _PAYLOAD_DATE/_PAYLOAD_NUMBER miss them entirely — without this, evidence
# chunks phrased this way scored as zero-payload and lost seats to question-echo chunks.
_PAYLOAD_RELDATE = re.compile(
    r"\b(?:last|next|this)\s+(?:week|month|year|weekend|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"spring|summer|fall|autumn|winter)\b"
    r"|\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"few|couple(?:\s+of)?)\s+(?:day|week|month|year)s?\s+ago\b"
    r"|\byesterday\b|\brecently\b|\bago\b",
    re.IGNORECASE)
# Spelled-out quantities ("three months", "twenty dollars") — same payload signal as
# _PAYLOAD_NUMBER/_PAYLOAD_CURRENCY, just not digit-encoded.
_NUM_WORD = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand"
)
_PAYLOAD_SPELLED_NUM = re.compile(
    rf"\b(?:{_NUM_WORD})(?:[\s-]+(?:{_NUM_WORD}))*\s+"
    r"(?:dollars?|bucks|cents?|euros?|pounds?|days?|weeks?|months?|years?|times?|percent|%)\b",
    re.IGNORECASE)
_RETARGET_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "for", "with", "from", "this",
    "that", "what", "how", "many", "who", "when", "where", "why", "which", "whose", "is",
    "are", "was", "were", "did", "does", "do", "i", "you", "he", "she", "it", "they", "we",
    "my", "your", "his", "her", "their", "our", "at", "by", "as", "be", "been", "has",
    "have", "had", "not", "no",
}



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
    retargeted: list[dict] = field(default_factory=list)       # chunk-retarget swaps/promotions
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
    "For a question about a specific timeframe or ordering ('initially', 'at first', "
    "'before X', 'when I started'): state the answer for THAT timeframe first, and only "
    "then note any later change — never lead with the current/latest state when the "
    "question asks about an earlier one.\n"
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
        self.last_retargeted: list[dict] = []

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

    def _source_chunks(self, base: str, known_idxs: set[int]) -> list[str]:
        """Every chunk episode belonging to `base` (e.g. `ep_sess1`). Production stores
        wire a `src_<item_id>` SOURCE parent with PART_OF edges from every chunk (see
        kg/ingest.py._write_parents) — that's authoritative and O(chunks-of-source). Tests
        (and any store without that parent) fall back to probing the `#cNNN` id space
        directly, same technique _expand_siblings uses for sibling existence checks."""
        item_id = base[len("ep_"):] if base.startswith("ep_") else base
        parent = f"src_{item_id}"
        kids = [nid for nid, _d in self.store.neighbors(parent, etypes={EdgeType.PART_OF},
                                                         direction="in")]
        if kids:
            return kids
        out: list[str] = []
        hi = max(known_idxs, default=0) + 50
        miss_streak = 0
        for i in range(hi + 1):
            cid = f"{base}#c{i:03d}"
            if self.store.get_node(cid):
                out.append(cid)
                miss_streak = 0
            else:
                miss_streak += 1
                if out and miss_streak > 10:
                    break
        return out

    def _chunk_text(self, eid: str) -> str:
        n = self.store.get_node(eid)
        return (n.raw_text or "") if n else ""

    def _payload_bonus(self, text: str) -> float:
        """Reward a chunk for carrying dates/numbers/currency amounts REGARDLESS of
        whether they overlap the question's own wording — a chunk that only restates the
        question has zero payload and must not out-score one that has overlap AND payload."""
        bonus = 0.0
        if _PAYLOAD_DATE.search(text):
            bonus += 4
        if _PAYLOAD_CURRENCY.search(text):
            bonus += 3
        if _PAYLOAD_NUMBER.search(text):
            bonus += 1
        if _PAYLOAD_RELDATE.search(text):
            bonus += 4
        if _PAYLOAD_SPELLED_NUM.search(text):
            bonus += 3
        return bonus

    def _lex_score(self, text: str, word_terms: set[str], digit_terms: set[str]) -> float:
        toks = set(_WORD.findall(text.lower()))
        return (len(toks & word_terms) + 3 * len(toks & digit_terms)
                + self._payload_bonus(text))

    def _retarget_source(self, cur: list[str], pool: list[str], seed_scores: dict,
                         word_terms: set[str] | None, digit_terms: set[str] | None) -> list[str]:
        """Refill one source's slots (len(cur) of them) by embedding seed rank, then
        (if lexical terms were supplied) swap in any unselected chunk of the same source
        that strictly beats a selected one on question-term overlap. The best-PPR-ranked
        (incumbent) chunk of the source is never displaced. Same count in and out."""
        n = len(cur)
        incumbent = cur[0]
        ranked = sorted(pool, key=lambda c: (-seed_scores.get(c, 0.0), c))
        picked = ranked[:n]
        if incumbent not in picked:
            picked = picked[: max(n - 1, 0)] + [incumbent]

        if word_terms is not None:
            lex = {c: self._lex_score(self._chunk_text(c), word_terms, digit_terms) for c in pool}
            picked_set = set(picked)
            for cand in sorted((c for c in pool if c not in picked_set), key=lambda c: -lex[c]):
                min_score = min(lex[c] for c in picked)
                tied = [c for c in picked if lex[c] == min_score]
                # a tie for worst protects the incumbent (it "survives ties"); if the
                # incumbent is the sole minimum it can still be strictly outscored below
                worst = (next(c for c in tied if c != incumbent)
                        if incumbent in tied and len(tied) > 1 else tied[0])
                if lex[cand] <= lex[worst]:
                    break   # sorted descending — no later candidate can beat `worst` either
                picked[picked.index(worst)] = cand
                picked_set.discard(worst)
                picked_set.add(cand)
        return picked

    def _retarget_chunks(self, selected: list[str], result: RetrievalResult) -> list[str]:
        """Query-side chunk retargeting (rag_retarget): the right SOURCE can win seats in
        _select_episodes while PPR's chunk-order picks the wrong CHUNK of it. Swaps only —
        same sources, same per-source slot counts, never adds or removes a seat. Defaults to
        a no-op (rag_retarget='off') so context stays byte-identical unless opted in."""
        mode = getattr(self.config, "rag_retarget", "off")
        if mode not in ("seed", "seed+lex"):
            return selected

        per_source: dict[str, list[str]] = {}
        for eid in selected:
            m = _CHUNK_ID.match(eid)
            if m:
                per_source.setdefault(m.group(1), []).append(eid)

        seed_scores = dict(getattr(result, "seed_scores", None) or {})
        word_terms = digit_terms = None
        if mode == "seed+lex":
            toks = _WORD.findall((result.query or "").lower())
            word_terms = {t for t in toks if t not in _RETARGET_STOP and not t.isdigit()
                          and len(t) > 2}
            digit_terms = {t for t in toks if t.isdigit()}

        picks: dict[str, list[str]] = {}
        for base, cur in per_source.items():
            idxs = {int(_CHUNK_ID.match(c).group(2)) for c in cur}
            pool = self._source_chunks(base, idxs)
            if len(pool) <= len(cur):
                picks[base] = cur
                continue
            picked = self._retarget_source(cur, pool, seed_scores, word_terms, digit_terms)
            picks[base] = picked
            if set(picked) != set(cur):
                self.last_retargeted.append({"kind": "retarget", "source": base,
                                             "from": cur, "to": picked})

        out: list[str] = []
        emitted: set[str] = set()
        for eid in selected:
            m = _CHUNK_ID.match(eid)
            if not m:
                out.append(eid)
                continue
            base = m.group(1)
            if base in emitted:
                continue
            emitted.add(base)
            out.extend(sorted(picks[base], key=lambda c: int(_CHUNK_ID.match(c).group(2))))
        return out

    def _promote_provenance(self, ctx_ids: list[str], selected: list[str],
                            facts: list[FactLine], query: str) -> list[str]:
        """rag_provenance_promote: pull a fact's source chunk (FactLine.episode_id) into
        context when the fact's src/dst entity names overlap the question terms, so the
        chunk a decisive fact was extracted from isn't left out just because it wasn't a
        sibling of a selected chunk. Only ever displaces the lowest-ranked expansion
        sibling (never an originally selected chunk); if none is displaceable, skipped."""
        if not getattr(self.config, "rag_provenance_promote", False):
            return ctx_ids
        toks = _WORD.findall((query or "").lower())
        terms = {t for t in toks if t not in _RETARGET_STOP and len(t) > 2}
        if not terms:
            return ctx_ids

        wanted: list[str] = []
        seen_w: set[str] = set()
        for f in facts:
            if not f.episode_id or f.episode_id in seen_w:
                continue
            names = _WORD.findall(f"{f.src} {f.dst}".lower())
            if set(names) & terms:
                seen_w.add(f.episode_id)
                wanted.append(f.episode_id)

        selected_set = set(selected)
        out = list(ctx_ids)
        ctx_set = set(out)
        for eid in wanted:
            if eid in ctx_set or not self.store.get_node(eid):
                continue
            expansion_only = [c for c in out if c not in selected_set]
            if not expansion_only:
                break   # nothing displaceable left — never touch an originally selected chunk
            victim = expansion_only[-1]
            out[out.index(victim)] = eid
            ctx_set.discard(victim)
            ctx_set.add(eid)
            self.last_retargeted.append({"kind": "provenance_promote",
                                         "displaced": victim, "promoted": eid})
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
        self.last_retargeted = []
        ep_ids = self._select_episodes(result.object_ids)
        ep_ids = self._retarget_chunks(ep_ids, result)
        ents = self.relevant_entities(result, ep_ids)
        facts = self.facts_for(ents, result.as_of)
        ctx_ids = self._expand_siblings(ep_ids)
        ctx_ids = self._promote_provenance(ctx_ids, ep_ids, facts, result.query)

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

    @staticmethod
    def _parse_message(msg) -> tuple[str, list]:
        ans, raw = "", []
        tc = getattr(msg.choices[0].message, "tool_calls", None) if msg.choices else None
        if tc and tc[0].function.name == "submit_answer":
            payload = json.loads(tc[0].function.arguments)
            ans = str(payload.get("answer", "")).strip()
            raw = payload.get("citations", [])
        elif msg.choices and msg.choices[0].message.content:
            ans = msg.choices[0].message.content.strip()
        return ans, raw

    @staticmethod
    def _finish_reason(msg) -> str | None:
        return getattr(msg.choices[0], "finish_reason", None) if msg.choices else None

    def answer(self, result: RetrievalResult) -> RagAnswer:
        with prof_span("query.build_context"):
            ep_ids, facts, blob = self.builder.build(result)
        base = RagAnswer(query=result.query, answer="", backend=self.name,
                         as_of=result.as_of, context_episodes=ep_ids,
                         facts=[f.render() for f in facts], object_ids=result.object_ids,
                         seeds=result.seeds, touched=sorted(result.subgraph),
                         retargeted=list(self.builder.last_retargeted))
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
            token_key = "max_completion_tokens"
        else:
            token_key = "max_tokens"
            kwargs["temperature"] = 0
        kwargs[token_key] = self.config.rag_max_tokens
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
        ans, raw = self._parse_message(msg)
        # A reasoning model can spend its ENTIRE completion budget on reasoning tokens and
        # never reach the submit_answer call — finish_reason="length" with no content/tool
        # call at all, not an API error, so the except above never sees it. One retry at
        # double the token cap before degrading to the extractive fallback.
        if not ans and self._finish_reason(msg) == "length":
            retry_kwargs = dict(kwargs)
            retry_kwargs[token_key] = kwargs[token_key] * 2
            try:
                with prof_span("query.llm_answer_retry"):
                    msg = call_with_backoff(
                        lambda: self.client.chat.completions.create(**retry_kwargs))
                self.meter.record("rag", self.config.rag_model, msg, label=result.query[:40])
                base.usage = self.meter.totals()
                ans, raw = self._parse_message(msg)
            except Exception as e:  # noqa: BLE001 — retry failed too, fall through below
                base.notes.append(f"retry after empty/length response failed: {e!r}")
            if not ans:
                base.answer = _extractive(self.store, result.query, ep_ids, facts)
                base.citations = ep_ids
                base.usage = self.meter.totals()
                base.notes.append(
                    f"empty answer after doubling token cap to {retry_kwargs[token_key]}, "
                    "used extractive fallback")
                return base
            base.notes.append(
                f"retried with doubled token cap ({retry_kwargs[token_key]}) after an empty "
                "response truncated by the length limit")
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
