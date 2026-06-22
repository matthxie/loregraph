"""kg — an LLM-traversable, directed knowledge graph over multimodal content.

See docs/ARCHITECTURE.md for the design this implements. The public surface is small:

    from kg import KnowledgeGraph
    g = KnowledgeGraph.open("store/")     # load or create a persisted graph
    g.ingest_object(...)                  # run the §6 ingestion pipeline on one object
    g.query("...")                        # 2-path retriever (§5)

Everything is pluggable: the LLM extractor (Haiku ⇄ offline heuristic) and the
embedder (sentence-transformers ⇄ hashing) auto-select based on what's available,
so the whole pipeline runs with or without an API key / model download.
"""
from .config import Config
from .graph import KnowledgeGraph

__all__ = ["Config", "KnowledgeGraph"]
__version__ = "0.1.0"
