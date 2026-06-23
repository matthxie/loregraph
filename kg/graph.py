"""KnowledgeGraph — the public facade that wires the pieces together.

    g = KnowledgeGraph.open("store/kg.db")
    g.ingest(load_corpus())
    g.build_communities()
    g.query("how are X and Y connected?")          # local → PPR seed-and-spread
    g.query("what are the main themes?")            # global → community map-reduce
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
        report = ing.ingest(items)
        return report

    def ingest_object(self, item: CorpusItem) -> IngestReport:
        return self.ingest([item])

    # ------------------------------------------------------------ communities
    def build_communities(self) -> int:
        return build_communities(self.store, self.embedder, self.config)

    # ----------------------------------------------------------------- query
    def query(self, text: str, mode: str = "auto", k: int | None = None):
        """Route + retrieve. mode: 'auto' | 'ppr' | 'bfs' | 'vector' | 'community'."""
        if mode == "auto":
            mode = "community" if (is_global_query(text)
                                   and self.store.nodes_of_type(NodeType.COMMUNITY)) else "ppr"
        if mode == "community":
            return CommunityRetriever(self.store, self.embedder, self.config).retrieve(
                text, k=k or 5)
        retriever = get_retriever(mode, self.store, self.embedder, self.canon, self.config)
        return retriever.retrieve(text, k=k)

    # ------------------------------------------------------------------- ask
    def ask(self, text: str, *, backend: str | None = None, k: int | None = None,
            max_steps: int | None = None, model: str | None = None, client=None):
        """Agentic LLM graph-traversal query (docs/ARCHITECTURE.md §5 reserved path).

        Hands an LLM read-only graph tools (seed-and-spread, keyword/vector search,
        neighbors, find_path, read_object, browse_themes) and lets it traverse to a cited
        answer. Falls back to a deterministic offline agent with no API key. Returns an
        AgentAnswer. `client=` injects a (possibly fake) Anthropic client for tests."""
        from .agent import get_agent
        cfg = self.config
        if backend or max_steps or model:
            cfg = replace(cfg, **{kk: vv for kk, vv in
                                  (("agent_backend", backend), ("agent_max_steps", max_steps),
                                   ("agent_model", model)) if vv})
        return get_agent(self.store, self.embedder, self.canon, cfg, client=client).run(text)

    # ----------------------------------------------------------------- helpers
    def explain(self, result: RetrievalResult, max_objects: int = 5) -> str:
        """Human-readable trace of a retrieval (uses the touched subgraph)."""
        lines = [f"query: {result.query!r}  mode={result.mode}",
                 f"seeds: {', '.join(result.seeds[:8])}"]
        for oid, score in result.objects[:max_objects]:
            n = self.store.get_node(oid)
            if n:
                lines.append(f"  [{score:.4f}] {oid}  {n.name}")
        return "\n".join(lines)

    def stats(self) -> dict:
        import os
        s = self.store.stats()
        # surface the live backends so a degraded offline run is never silent
        agent = "claude" if (self.config.agent_backend in ("auto", "claude")
                             and os.environ.get("ANTHROPIC_API_KEY")) else "offline"
        s["backends"] = {"extractor": self.extractor.name, "embedder": self.embedder.name,
                         "agent": agent}
        return s
