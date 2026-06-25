"""KnowledgeGraph — the public facade that wires the pieces together.

    g = KnowledgeGraph.open("store/kg.db")
    g.ingest(load_longmemeval("small"))
    g.build_communities()
    g.query("how are X and Y connected?")           # local → PPR seed-and-spread
    g.query("what are the main themes?")             # global → community map-reduce
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
from .rag import RagAnswer, get_answerer
from .retrieval import RetrievalResult, get_retriever
from .store import GraphStore


class KnowledgeGraph:
    def __init__(self, store: GraphStore, config: Config | None = None):
        self.config = config or store.config
        self.store = store
        self.embedder = get_embedder(self.config)
        self.extractor = get_extractor(self.config)
        self.canon = Canonicalizer(store, self.embedder, self.config)

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
        retriever = get_retriever(mode, self.store, self.embedder, self.canon, self.config)
        return retriever.retrieve(text, k=k, as_of=as_of)

    # ------------------------------------------------------------------- ask
    def ask(self, text: str, *, backend: str | None = None, k: int | None = None,
            as_of: str | None = None, model: str | None = None, client=None) -> RagAnswer:
        """Graph-RAG answer (docs/ARCHITECTURE.md §5): PPR retrieves a context blob (top
        episodes + currently-valid facts), then ONE LLM call answers over it with
        citations. The LLM never traverses. Falls back to a deterministic offline
        answerer with no API key. `client=` injects a (fake) Anthropic client for tests."""
        cfg = self.config
        overrides = {kk: vv for kk, vv in
                     (("rag_backend", backend), ("rag_model", model)) if vv}
        if overrides:
            cfg = replace(cfg, **overrides)
        return get_answerer(self.store, self.embedder, self.canon, cfg,
                            client=client).run(text, k=k, as_of=as_of)

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
        import os
        s = self.store.stats()
        ans = "claude" if (self.config.rag_backend in ("auto", "claude")
                           and os.environ.get("ANTHROPIC_API_KEY")) else "offline"
        s["backends"] = {"extractor": self.extractor.name, "embedder": self.embedder.name,
                         "answerer": ans}
        return s
