"""kg — an LLM-traversable, directed knowledge graph over multimodal content.

See docs/ARCHITECTURE.md for the design this implements. The public surface is small:

    from kg import KnowledgeGraph
    g = KnowledgeGraph.open("store/")     # load or create a persisted graph
    g.ingest_object(...)                  # run the §6 ingestion pipeline on one object
    g.query("...")                        # 2-path retriever (§5)

Everything is pluggable: the LLM extractor (Haiku ⇄ offline heuristic) and the
embedder (sentence-transformers ⇄ hashing) auto-select based on what's available,
so the whole pipeline runs with or without an API key / model download.

On import we load a project-root ``.env`` (if present) so ``ANTHROPIC_API_KEY``
and friends are available before the extractor auto-selects. Real environment
variables always win — the file only fills in what isn't already set — and a
missing python-dotenv is non-fatal (the pipeline just stays offline).
"""
from pathlib import Path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return  # optional dep; without it we rely on the real environment
    # Repo root is one level up from this package; load that .env explicitly so
    # it works regardless of the current working directory. override=False keeps
    # an already-exported var ahead of the file.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


_load_dotenv()

from .agent import AgentAnswer
from .config import Config
from .graph import KnowledgeGraph

__all__ = ["AgentAnswer", "Config", "KnowledgeGraph"]
__version__ = "0.1.0"
