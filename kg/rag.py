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
import re
from dataclasses import dataclass, field

from .backoff import call_with_backoff
from .canonicalize import Canonicalizer
from .config import Config
from .embedders import Embedder
from .facts import FactIndex, FactLine
from .llm_client import RAG_OPENAI_DEFAULT, llm_available, make_client, resolve_model
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


def _when_with_delta(created_at: str | None, as_of: str | None) -> str:
    """Render an episode's date for a context line: "(YYYY-MM-DD)", or, when both the
    episode date and the question anchor (`as_of`) parse, "(YYYY-MM-DD, N days before the
    question)" / "(..., same day as the question)" / "(..., N days after the question)".
    Compares the date part only. Falls back to the bare date on any missing/unparseable
    input."""
    when = (created_at or "")[:10]
    if not when or not as_of:
        return f"({when})"
    from datetime import date
    try:
        ep = date.fromisoformat(when)
        anchor = date.fromisoformat(str(as_of)[:10])
    except ValueError:
        return f"({when})"
    delta = (ep - anchor).days
    if delta == 0:
        return f"({when}, same day as the question)"
    n = abs(delta)
    unit = "day" if n == 1 else "days"
    rel = "after" if delta > 0 else "before"
    return f"({when}, {n} {unit} {rel} the question)"


# --------------------------------------------------------------------------- #
# In-text relative-date resolution (config.rag_resolve_reldates)
# --------------------------------------------------------------------------- #
# An episode saying "last week I attended the workshop" places the EVENT on a different
# date than the episode — the header date/delta can't express that, and the reader
# routinely anchors on the episode date instead. Resolve each relative phrase against the
# episode's own date and annotate it inline ("last week [≈ 2023-03-20] I attended ..."),
# so event dates become absolute where the text states them relatively. Exact phrases
# (yesterday, last Saturday) get "=", fuzzy ones (two months ago, last month) get "≈".
_ANNOT_NUM_WORD = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                   "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                   "twelve": 12, "couple": 2, "few": 3}
_ANNOT_WEEKDAY = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                  "friday": 4, "saturday": 5, "sunday": 6}
_REL_ANNOT = re.compile(
    r"\byesterday\b"
    r"|\blast\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"couple|few)\s+(?:of\s+)?(?:day|week|month|year)s?\s+ago\b"
    r"|\blast\s+(?:week|month|year)\b",
    re.IGNORECASE)


def _resolve_rel_phrase(phrase: str, anchor: "datetime.date"):
    """Resolve one _REL_ANNOT match against `anchor` → (date, exact: bool) or None."""
    from datetime import timedelta
    p = phrase.lower()
    if p == "yesterday":
        return anchor - timedelta(days=1), True
    m = re.match(r"last\s+(\w+)$", p)
    if m and m.group(1) in _ANNOT_WEEKDAY:
        back = (anchor.weekday() - _ANNOT_WEEKDAY[m.group(1)]) % 7 or 7
        return anchor - timedelta(days=back), True
    if m:                                       # last week/month/year — fuzzy midpoint
        days = {"week": 7, "month": 30, "year": 365}[m.group(1)]
        return anchor - timedelta(days=days), False
    m = re.match(r"(\d+|\w+)\s+(?:of\s+)?(day|week|month|year)s?\s+ago$", p)
    if m:
        raw = m.group(1)
        n = int(raw) if raw.isdigit() else _ANNOT_NUM_WORD.get(raw)
        if n is None:
            return None
        days = n * {"day": 1, "week": 7, "month": 30, "year": 365}[m.group(2)]
        exact = m.group(2) == "day" and (raw.isdigit() or raw in
                                         ("a", "an", "one", "two", "three"))
        return anchor - timedelta(days=days), exact
    return None


def _annotate_relative_dates(text: str, created_at: str | None) -> str:
    """Annotate relative-date phrases in episode text with the absolute date they resolve
    to against the episode's own date. No-op when the episode date doesn't parse."""
    from datetime import date
    try:
        anchor = date.fromisoformat((created_at or "")[:10])
    except ValueError:
        return text

    def sub(m):
        r = _resolve_rel_phrase(m.group(0), anchor)
        if r is None:
            return m.group(0)
        d, exact = r
        return f"{m.group(0)} [{'=' if exact else '≈'} {d.isoformat()}]"

    return _REL_ANNOT.sub(sub, text)


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
    fact_rows: list[dict] = field(default_factory=list)        # structured §3 Fact objects
    context_text: str = ""                                     # the §5.1 block the LLM read
    object_ids: list[str] = field(default_factory=list)        # PPR ranking (eval seam)
    ppr_pool: list = field(default_factory=list)               # (ep_id, raw PPR score) pool
    seeds: list[str] = field(default_factory=list)
    touched: list[str] = field(default_factory=list)           # every node in the PPR subgraph
    retargeted: list[dict] = field(default_factory=list)       # chunk-retarget swaps/promotions
    events: list[dict] = field(default_factory=list)           # enumeration scaffold (rag_answer_events)
    usage: dict = field(default_factory=dict)                  # token/cost (empty offline)
    steps: int = 1            # retrieve-then-read = ONE answer call (no per-hop loop)
    stopped: str = "answered"
    trace: list = field(default_factory=list)                  # no tool trace (RAG, not agentic)
    notes: list[str] = field(default_factory=list)


@dataclass
class SearchHit:
    """One retrieved memory, feed-ready: the episode chunk's id, relevance score,
    timestamp, and text."""
    episode_id: str
    score: float                 # retrieval score (0.0 for provenance/sibling additions)
    when: str = ""               # episode created_at (ISO), "" if unknown
    name: str = ""
    text: str = ""


@dataclass
class SearchResult:
    """What `KnowledgeGraph.search()` returns: the same evidence ask() would hand its
    answering LLM, but structured for direct display (a feed, search results page)
    instead of prompt-formatted. `.context` keeps the exact prompt blob for callers
    who want to run their own LLM over it."""
    query: str
    hits: list[SearchHit] = field(default_factory=list)   # context order (relevance-led)
    facts: list[str] = field(default_factory=list)         # rendered fact lines
    fact_rows: list[dict] = field(default_factory=list)    # structured §3 Fact objects,
    #                                                        same facts in the same order
    lane: str = ""
    as_of: str | None = None
    context: str = ""


class Searcher:
    """ask() minus the answering LLM: hybrid-retrieve (route → PPR → augment → rerank)
    then assemble the evidence, returned as structured hits. Fully offline — needs no
    OPENAI_API_KEY (the only model involved is the local cross-encoder, which degrades
    gracefully when unavailable)."""

    def __init__(self, store: GraphStore, embedder: Embedder, canon: Canonicalizer,
                 config: Config):
        self.store = store
        self.config = config
        self.retriever = HybridRetriever(store, embedder, canon, config)
        self.builder = ContextBuilder(store, config)

    def run(self, query: str, k: int | None = None, as_of: str | None = None,
            rerank: bool | None = None, mmr_lambda: float | None = None,
            since: str | None = None, until: str | None = None) -> SearchResult:
        if not query or not query.strip():
            return SearchResult(query=query, as_of=as_of)
        result = self.retriever.retrieve(query, k=k or self.config.top_k, as_of=as_of,
                                         rerank=rerank, mmr_lambda=mmr_lambda,
                                         since=since, until=until)
        ep_ids, facts, blob = self.builder.build(result)
        scores = dict(result.objects)
        hits = []
        for eid in ep_ids:
            n = self.store.get_node(eid)
            if not n:
                continue
            text = _WS.sub(" ", (n.raw_text or n.description or n.name or "")).strip()
            hits.append(SearchHit(episode_id=eid, score=float(scores.get(eid, 0.0)),
                                  when=n.created_at or "", name=n.name, text=text))
        return SearchResult(query=query, hits=hits,
                            facts=[f.render() for f in facts],
                            fact_rows=[f.to_row() for f in facts],
                            lane=getattr(result, "lane", ""), as_of=as_of, context=blob)


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
    "compute the day difference step by step (months and remaining days) before answering. "
    "Each episode header may carry a parenthetical time difference (e.g. '(2023-03-05, 22 "
    "days before the question)'): it states when that EPISODE's conversation happened "
    "relative to the question, as an input for your arithmetic — it is never itself the "
    "answer. Copy a difference only from the episode where the asked-about event actually "
    "occurs, convert it to the unit the question asks for (months, weeks), and remember an "
    "event mentioned in an episode may have happened on a different date than the episode.\n"
    "When the question asks about the user's own cost, amount, or situation, use figures "
    "the USER stated about their specific case (\"I was told the repair would cost $80\") — "
    "the assistant's generic price ranges, typical fares, or estimates in the same "
    "conversations are facts about the world, not about the user. Prefer a user-stated "
    "figure over a generic one even when the assistant disputed it. If a needed operand "
    "exists only as a generic figure, say the information is insufficient rather than "
    "computing from it.\n"
    "Verify the exact subject of the question appears in the context. If the context only "
    "contains a similar but different item (e.g. asked about antique maps but the context "
    "only has antique globes), say you don't have that information instead of substituting "
    "the similar item. This applies especially to place names and dates: if asked about "
    "city/venue X and the context only covers a different city/venue Y, say the information "
    "is not available — do not answer about Y. EXCEPTION: when the context has an item of a "
    "closely related category that matches the question's person and timeframe (asked about "
    "artwork; the context has a gift of a hand-painted vase from the same person on the same "
    "date), give it as the likely answer and note the wording differs — a category near-miss "
    "with the right person and date is an answer, not a refusal.\n"
    "Commit to the single best-supported answer. When the evidence is indirect (relative "
    "dates to resolve, two statements to combine, an amount to derive), state your "
    "assumption and answer anyway — do not refuse because the answer requires combining "
    "statements. Say the information is unavailable ONLY when nothing in the context bears "
    "on the question's subject.\n"
    "For advice or recommendation questions, anchor every suggestion to the specific items, "
    "purchases, plans, or preferences the user stated in the context episodes — build on "
    "what they already have or said, never give generic advice a stranger could get.\n"
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

# Aggregation/temporal variant (config.rag_answer_events): identical call shape, but the
# schema REQUIRES `events` — every question-relevant event enumerated with its date and
# quantity — listed BEFORE the answer. The reader's dominant failure on "how many / how
# much / how long" questions is emitting a total it never enumerated; a required list
# field makes the enumeration the path of least resistance instead of an instruction the
# model may skip. `events` is scaffolding: it is surfaced on RagAnswer.events for
# inspection but never judged and never fed back into another call.
_ANSWER_TOOL_EVENTS = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": ("Submit the final answer grounded in the provided context. FIRST "
                        "fill `events` with EVERY event/statement in the context that "
                        "matches the question (one entry per distinct event, with its "
                        "date and any quantity), THEN derive the answer from that list "
                        "alone — count/sum/compare the listed events, never a number you "
                        "did not enumerate."),
        "parameters": {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "description": ("every context event matching the question; leave "
                                    "empty only if none exist"),
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string",
                                     "description": "when it happened (episode date if "
                                                    "the text is relative)"},
                            "description": {"type": "string"},
                            "quantity": {"type": "string",
                                         "description": "amount/count/duration if any"},
                            "source": {"type": "string",
                                       "enum": ["user", "assistant"],
                                       "description": ("who stated it: 'user' if the user "
                                                       "said it about their own situation "
                                                       "(\"I was told the repair costs $80\"), "
                                                       "'assistant' for the assistant's "
                                                       "generic prices, typical ranges, or "
                                                       "estimates")},
                        },
                        "required": ["date", "description", "source"],
                    },
                },
                "calculation": {"type": "string",
                                "description": ("the arithmetic, written out step by step "
                                                "from the listed events (counts, sums, or "
                                                "date differences as months+days); write "
                                                "'none' if the answer needs no arithmetic. "
                                                "For the user's own costs/amounts, use "
                                                "operands from source='user' events only — "
                                                "if a needed operand has no source='user' "
                                                "event, answer that the information is "
                                                "insufficient instead of computing from "
                                                "source='assistant' figures")},
                "answer": {"type": "string",
                           "description": "derived ONLY from the events and calculation above"},
                "citations": {"type": "array", "items": {"type": "string"},
                              "description": "episode ids you used, e.g. ep_2"},
            },
            "required": ["events", "calculation", "answer", "citations"],
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
        self._ce = None   # lazy cross-encoder for rag_retarget="ce" (local, $0)

    def _ce_scores(self, query: str, chunk_ids: list[str]) -> dict[str, float] | None:
        """Cross-encoder relevance of each chunk to the question (rag_retarget='ce').
        Reuses the same local ms-marco model the retriever's rerank lane uses; returns
        None when the model isn't available so the caller can fall back to seed order."""
        if self._ce is None:
            from .rerank import CrossEncoderReranker
            self._ce = CrossEncoderReranker(self.config.rerank_model)
        if not self._ce.available:
            return None
        pairs = [(cid, self._chunk_text(cid)[:1000]) for cid in chunk_ids]
        ranked = self._ce.rerank(query, pairs, len(pairs))
        # rerank returns ids in score order; encode order as descending pseudo-scores
        return {cid: float(len(ranked) - i) for i, cid in enumerate(ranked)}

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
                    out.append(FactLine.from_edge(self.store, src_id, dst_id, data))
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
        if mode not in ("seed", "seed+lex", "ce", "ce+seed"):
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
            scores = seed_scores
            if mode == "ce+seed":
                # Signal-diverse refill: CE, seed-embedding, and lexical question-overlap
                # each find answer chunks the other two miss (sweep 2026-07-06: CE finds
                # the cat-name chunk, lex finds the UCLA chunk, seed neither — and lex as
                # a swap PASS evicts the CE pick, so it must live in the blend, not after
                # it). Rank each chunk by its BEST rank across the three signals, ties
                # broken toward the CE ordering.
                toks = _WORD.findall((result.query or "").lower())
                wt = {t for t in toks if t not in _RETARGET_STOP and not t.isdigit()
                      and len(t) > 2}
                dt = {t for t in toks if t.isdigit()}
                lex = {c: self._lex_score(self._chunk_text(c), wt, dt) for c in pool}
                ce = self._ce_scores(result.query, pool) or {}

                def rank_of(sc):
                    order = sorted(pool, key=lambda c: (-sc.get(c, 0.0), c))
                    return {c: i for i, c in enumerate(order)}
                rc, rs, rl = rank_of(ce), rank_of(seed_scores), rank_of(lex)
                scores = {c: -min(rc[c], rs[c], rl[c]) - 0.001 * rc[c] for c in pool}
            elif mode == "ce":
                # question<->chunk relevance from the local cross-encoder — a direct
                # "which chunk of this session answers THIS question" signal, stabler
                # than PPR chunk order (which churns with any upstream ranking change —
                # the cat/Luna failure) and than the payload regexes (which favour any
                # digit-dense chunk). Falls back to seed order if the model can't load.
                scores = self._ce_scores(result.query, pool) or seed_scores
            picked = self._retarget_source(cur, pool, scores, word_terms, digit_terms)
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
        # §7.3 hard bound: sibling expansion and provenance promotion re-inject
        # episodes AFTER the retriever's since/until filter — a windowed request must
        # never see (or have its answer read) an out-of-window episode. Facts are
        # deliberately not windowed; the bound is on the returned EPISODES.
        since, until = getattr(result, "window", None) or (None, None)
        if since or until:
            def _in_window(eid: str) -> bool:
                n = self.store.get_node(eid)
                d = ((n.created_at or n.ingested_at or "") if n else "")[:10]
                if not d:
                    return False
                return ((not since or d >= since[:10])
                        and (not until or d <= until[:10]))
            ctx_ids = [eid for eid in ctx_ids if _in_window(eid)]

        lines = [f"QUESTION: {result.query}",
                 f"AS-OF: {result.as_of or 'now (current view)'}", ""]
        lines.append("EPISODES (evidence; cite by id):")
        if ctx_ids:
            for eid in ctx_ids:
                n = self.store.get_node(eid)
                if not n:
                    continue
                when = _when_with_delta(n.created_at, result.as_of)
                snippet = self._snippet(n, self.config.rag_episode_chars)
                if getattr(self.config, "rag_resolve_reldates", False):
                    snippet = _annotate_relative_dates(snippet, n.created_at)
                lines.append(f"[{eid}] {when} {n.name}: {snippet}")
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


def strip_citations(text: str, ids) -> str:
    """Remove the given episode-id citations from answer text (PROTOCOL §3.12: invalid
    citations are dropped from `citations` AND stripped from the answer's text — the
    answerer may not display invented evidence either).

    Only EPISODE-ID-SHAPED tokens (_EP_ID) are removed: the model's citations array is
    free-form, and deleting an arbitrary dropped string (an entity name, a year) would
    mangle valid prose. Bracketed forms ("[ep_x]", "[ep_a, ep_x]") lose just the id.
    Cleanup is scoped to the removal sites via a marker — the rest of the answer's
    formatting (markdown line breaks, aligned quotes) is untouched."""
    to_strip = [cid for cid in (ids or [])
                if isinstance(cid, str) and _EP_ID.fullmatch(cid)]
    if not text or not to_strip:
        return text
    mark = "\x00"
    text = text.replace(mark, "")
    for cid in to_strip:
        text = re.sub(rf"(?<![A-Za-z0-9_#]){re.escape(cid)}(?![A-Za-z0-9_#])",
                      mark, text)
    text = re.sub(rf"\[\s*{mark}\s*(?:,\s*{mark}\s*)*\]", mark, text)  # emptied [..]
    text = re.sub(rf"\[\s*(?:{mark}\s*,\s*)+", "[", text)  # "[×, ×, ep_a]" → "[ep_a]"
    text = re.sub(rf",\s*{mark}\s*(?=[,\]])", "", text)  # "[ep_a, ×]" / "[a, ×, b]"
    text = re.sub(rf"\s*{mark}\s*([.,;:!?])", r"\1", text)  # "× ." → "."
    text = re.sub(rf"\s*{mark}\s*", " ", text)           # what remains → one space
    return text.strip()


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
    def _parse_message(msg) -> tuple[str, list, list]:
        ans, raw, events = "", [], []
        tc = getattr(msg.choices[0].message, "tool_calls", None) if msg.choices else None
        if tc and tc[0].function.name == "submit_answer":
            payload = json.loads(tc[0].function.arguments)
            ans = str(payload.get("answer", "")).strip()
            raw = payload.get("citations", [])
            ev = payload.get("events", [])
            events = [e for e in ev if isinstance(e, dict)] if isinstance(ev, list) else []
        elif msg.choices and msg.choices[0].message.content:
            ans = msg.choices[0].message.content.strip()
        return ans, raw, events

    def _answer_tool(self, result: RetrievalResult) -> dict:
        """The plain submit_answer schema, or the events-enumeration variant when
        rag_answer_events covers this query's lane ("all", or "lanes" + a routed lane in
        rag_answer_events_lanes). Default "off" keeps the schema byte-identical to today."""
        mode = getattr(self.config, "rag_answer_events", "off")
        if mode == "all":
            return _ANSWER_TOOL_EVENTS
        if mode == "lanes":
            lanes = set(getattr(self.config, "rag_answer_events_lanes", ()))
            if getattr(result, "lane", "single") in lanes:
                return _ANSWER_TOOL_EVENTS
        return _ANSWER_TOOL

    @staticmethod
    def _finish_reason(msg) -> str | None:
        return getattr(msg.choices[0], "finish_reason", None) if msg.choices else None

    def answer(self, result: RetrievalResult) -> RagAnswer:
        with prof_span("query.build_context"):
            ep_ids, facts, blob = self.builder.build(result)
        base = RagAnswer(query=result.query, answer="", backend=self.name,
                         as_of=result.as_of, context_episodes=ep_ids,
                         facts=[f.render() for f in facts],
                         fact_rows=[f.to_row() for f in facts],
                         context_text=blob, object_ids=result.object_ids,
                         seeds=result.seeds, touched=sorted(result.subgraph),
                         retargeted=list(self.builder.last_retargeted))
        # explicit rag_model wins; unset resolves per provider (openai → gpt-5-mini)
        model = resolve_model(self.config.rag_model, openai_default=RAG_OPENAI_DEFAULT)
        # gpt-5 / o-series models reject `max_tokens` (they want max_completion_tokens)
        # and any non-default temperature; 4o-era models keep the old params. Getting this
        # wrong would not crash loudly — the except below silently degrades every answer to
        # the offline extractive path — so the split must live here.
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": _RAG_SYS},
                {"role": "user", "content": blob},
            ],
            "tools": [self._answer_tool(result)],
            "tool_choice": {"type": "function", "function": {"name": "submit_answer"}},
        }
        if (model or "").startswith(("gpt-5", "o1", "o3", "o4")):
            token_key = "max_completion_tokens"
        else:
            token_key = "max_tokens"
            kwargs["temperature"] = 0
        kwargs[token_key] = self.config.rag_max_tokens
        try:
            with prof_span("query.llm_answer"):
                msg = call_with_backoff(lambda: self.client.chat.completions.create(**kwargs))
            self.meter.record("rag", model or "cli-default", msg, label=result.query[:40])
        except Exception as e:  # noqa: BLE001 — degrade to the offline synthesis, never crash
            base.answer = _extractive(self.store, result.query, ep_ids, facts)
            base.citations = ep_ids
            base.usage = self.meter.totals()
            base.notes.append(f"api error, used extractive fallback: {e!r}")
            return base
        base.usage = self.meter.totals()
        ans, raw, base.events = self._parse_message(msg)
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
                self.meter.record("rag", model or "cli-default", msg, label=result.query[:40])
                base.usage = self.meter.totals()
                ans, raw, base.events = self._parse_message(msg)
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
        # §3.12 validation universe: the context's episode ids PLUS the grounding
        # episode ids of the context's facts — both visibly appear in the rendered
        # block the answerer read, so citing either is legitimate evidence.
        universe = ep_ids + [f.episode_id for f in facts if f.episode_id]
        kept, dropped = _validate(raw, universe)
        # §3.12: dropped citations leave the answer TEXT too, not just the list —
        # the scrape above must run first or a citation-less tool reply loses its ids.
        base.answer = strip_citations(ans, dropped) or "(no answer produced)"
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
        """Live-only: an OpenAIAnswerer over an injected client, else the active provider's
        client from the spine (make_client). There is no offline backend — without a provider
        (and no injected client) we raise, rather than silently degrade to a fake answer."""
        if client is not None:
            return OpenAIAnswerer(self.store, self.config, self.builder, client=client)
        if llm_available():
            client = make_client()
            if client is not None:
                return OpenAIAnswerer(self.store, self.config, self.builder, client=client)
        raise RuntimeError(
            "No LLM provider available (set KG_LLM / OPENAI_API_KEY, or run codex login). "
            "The query/answer path is live-only. kg auto-reads a project-root .env, or "
            "inject a client: get_answerer(..., client=fake).")

    def run(self, query: str, k: int | None = None, as_of: str | None = None,
            rerank: bool | None = None, mmr_lambda: float | None = None,
            since: str | None = None, until: str | None = None) -> RagAnswer:
        if not query or not query.strip():
            return RagAnswer(query=query, answer="(empty query)",
                             backend=self._backend.name, as_of=as_of)
        result = self.retriever.retrieve(query, k=k or self.config.top_k, as_of=as_of,
                                         rerank=rerank, mmr_lambda=mmr_lambda,
                                         since=since, until=until)
        ans = self._backend.answer(result)
        ans.ppr_pool = list(getattr(result, "ppr_pool", []) or [])
        ans.lane = getattr(result, "lane", "")            # surface the routed lane
        ans.rerank_active = self.retriever.rerank_active
        return ans


def get_answerer(store, embedder, canon, config, *, client=None) -> RagAnswerer:
    return RagAnswerer(store, embedder, canon, config, client=client)
