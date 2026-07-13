"""KnowledgeGraph — the public facade that wires the pieces together.

    g = KnowledgeGraph.open("store/kg.db")
    g.ingest(load_longmemeval("small"))
    g.build_communities()
    g.query("how are X and Y connected?")           # local → PPR seed-and-spread
    g.query("what are the main themes?")             # global → community map-reduce
    g.search("Becky")                                # ranked memories, no LLM (feeds/UI)
    g.ask("where does Becky live?")                  # PPR-retrieve → context → 1 LLM answer
    g.ask("where did Becky live?", as_of="2022")     # point-in-time (as-of-T) retrieval
    g.save()
"""
from __future__ import annotations

from dataclasses import replace

from .canonicalize import Canonicalizer
from .communities import CommunityRetriever, build_communities, is_global_query
from .config import Config
from .corpus import CorpusItem
from .embedders import get_embedder
from .extractors import get_extractor
from .ingest import IngestReport, Ingestor
from .models import NodeType
from .rag import RagAnswer, SearchResult, Searcher, get_answerer
from .retrieval import RetrievalResult, get_retriever
from .store import GraphStore


class KnowledgeGraph:
    def __init__(self, store: GraphStore, config: Config | None = None):
        self.config = config or store.config
        self.store = store
        self.embedder = get_embedder(self.config)
        self.extractor = get_extractor(self.config)
        self.canon = Canonicalizer(store, self.embedder, self.config)
        # Long-lived retrieval state: retrievers/answerers are hoisted here so their
        # warm caches (BM25 corpus, projection, PPR operator) survive across calls
        # instead of being rebuilt per query. Staleness is handled inside (they key
        # off store.version / store.episode_version), so ingesting more data is safe.
        self._retrievers: dict[str, object] = {}
        self._answerer = None
        self._answerer_key: tuple | None = None
        self._searcher: Searcher | None = None

    # ------------------------------------------------------------------ open
    @classmethod
    def open(cls, path: str, config: Config | None = None) -> "KnowledgeGraph":
        config = config or Config.default()
        store = GraphStore.open(path, config=config)
        return cls(store, config)

    def save(self) -> None:
        self.store.save()

    # ---------------------------------------------------------------- ingest
    def ingest(self, items: list[CorpusItem]) -> IngestReport:
        ing = Ingestor(self.store, self.extractor, self.embedder, self.canon, self.config)
        return ing.ingest(items)

    def ingest_object(self, item: CorpusItem) -> IngestReport:
        return self.ingest([item])

    # ------------------------------------------------------------ communities
    def build_communities(self) -> int:
        return build_communities(self.store, self.embedder, self.config)

    # ---------------------------------------------------------------- forget
    def forget(self, secret: str, *, dry_run: bool = False, escalate: bool = True,
               client=None):
        """Erase a piece of information from memory (kg/forget.py): exhaustively sweep
        every chunk for it, redact the matched sentences in place (the rest of each
        turn survives), retract the facts/mentions/tags derived from the removed text,
        invalidate orphans, and loop until a re-sweep finds nothing. Returns an
        EraseReport (see its .summary()).

        `escalate=True` (default) adds LLM steps when a client/key is available:
        paraphrase confirmation for fuzzy hits, a single-chunk re-extract diff for
        artifact attribution, and a final inference audit that escalates to whole-chunk
        tombstones if the secret is still reconstructable from retrieval. Deterministic
        gates run first either way; without a key the erase is fully offline
        (paraphrased restatements are reported as `unconfirmed`, never silently kept).

        `dry_run=True` reports what WOULD be erased without mutating anything.
        Mutations are in memory until `save()`. Ingest caches and raw session logs are
        outside the store and must be purged separately."""
        from .forget import Eraser
        if client is None and escalate:
            from .llm_client import llm_available, make_client
            client = make_client() if llm_available() else None
        eraser = Eraser(self.store, self.embedder, self.canon, self.config,
                        extractor=self.extractor, client=client)
        return eraser.erase(secret, dry_run=dry_run, escalate=escalate)

    # ----------------------------------------------------------------- query
    def query(self, text: str, mode: str = "auto", k: int | None = None,
              as_of: str | None = None):
        """Route + retrieve. mode: 'auto' | 'ppr' | 'bfs' | 'vector' | 'community'.
        `as_of` (ISO date/year) retrieves the world as it was at that time (facts whose
        valid window contained it); default None = the current view."""
        if mode == "auto":
            mode = "community" if (is_global_query(text)
                                   and self.store.nodes_of_type(NodeType.COMMUNITY)) else "ppr"
        if mode == "community":
            return CommunityRetriever(self.store, self.embedder, self.config).retrieve(
                text, k=k or 5)
        retriever = self._retrievers.get(mode)
        if retriever is None:
            retriever = self._retrievers[mode] = get_retriever(
                mode, self.store, self.embedder, self.canon, self.config)
        return retriever.retrieve(text, k=k, as_of=as_of)

    # ---------------------------------------------------------------- search
    def search(self, text: str, *, k: int | None = None,
               as_of: str | None = None) -> SearchResult:
        """Everything ask() does EXCEPT the answering LLM call: the hybrid retriever
        routes the question and ranks the relevant memories, and the context builder
        assembles the same evidence the answerer would see — returned as structured
        hits (episode id, score, time, text) plus the relevant facts, search-engine
        style, for feeds/UI. `.context` carries the exact prompt blob ask()'s LLM
        would read. Fully offline: no OPENAI_API_KEY needed."""
        if self._searcher is None:
            self._searcher = Searcher(self.store, self.embedder, self.canon,
                                      self.config)
        return self._searcher.run(text, k=k, as_of=as_of)

    # ------------------------------------------------------------------- ask
    def ask(self, text: str, *, backend: str | None = None, k: int | None = None,
            as_of: str | None = None, model: str | None = None, client=None) -> RagAnswer:
        """Graph-RAG answer (docs/ARCHITECTURE.md §5): the hybrid retriever routes the
        question, augments state/evolution lanes with fact-bearing episodes, and reranks the
        hard lanes, then ONE LLM call answers over the context with citations. The LLM never
        traverses. The router reads the question TEXT only (never benchmark metadata —
        see tests/test_no_oracle.py). Live-only: needs
        OPENAI_API_KEY. `client=` injects a (fake) OpenAI client for tests."""
        cfg = self.config
        overrides = {kk: vv for kk, vv in
                     (("rag_backend", backend), ("rag_model", model)) if vv}
        if overrides:
            cfg = replace(cfg, **overrides)
        if client is not None:   # injected (test) client → no caching, exact old semantics
            return get_answerer(self.store, self.embedder, self.canon, cfg,
                                client=client).run(text, k=k, as_of=as_of)
        akey = (cfg.rag_backend, cfg.rag_model)
        if self._answerer is None or self._answerer_key != akey:
            self._answerer = get_answerer(self.store, self.embedder, self.canon, cfg)
            self._answerer_key = akey
        return self._answerer.run(text, k=k, as_of=as_of)

    # ----------------------------------------------------------------- helpers
    def explain(self, result: RetrievalResult, max_objects: int = 5) -> str:
        lines = [f"query: {result.query!r}  mode={result.mode}"
                 + (f"  as_of={result.as_of}" if result.as_of else ""),
                 f"seeds: {', '.join(result.seeds[:8])}"]
        for oid, score in result.objects[:max_objects]:
            n = self.store.get_node(oid)
            if n:
                lines.append(f"  [{score:.4f}] {oid}  {n.name}")
        return "\n".join(lines)

    def stats(self) -> dict:
        from .llm_client import current_provider, llm_available
        s = self.store.stats()
        ans = (current_provider()["kind"] if llm_available()
               else "unavailable (no LLM provider)")
        s["backends"] = {"extractor": self.extractor.name, "embedder": self.embedder.name,
                         "answerer": ans}
        return s
