"""kg — an episodic, bi-temporal knowledge graph over multimodal content.

See docs/ARCHITECTURE.md for the design this implements. The public surface is small:

    from kg import KnowledgeGraph
    g = KnowledgeGraph.open("store/")     # load or create a persisted graph
    g.ingest_object(...)                  # run the §6 ingestion pipeline on one entry
    g.query("...")                        # 2-path retriever (§5)
    g.ask("...")                          # PPR-retrieve → context → one LLM answer (§5)

The pipeline is LIVE-ONLY: extraction uses gpt-4o-mini and embeddings use a local
sentence-transformers model (BAAI/bge-small). The old offline heuristic/hashing
backends were removed. Deterministic tests/demos stub the LLM with a ScriptedExtractor
and inject a fake OpenAI client; the embedder runs the real (local, deterministic)
model.

On import we load a project-root ``.env`` (if present) so ``OPENAI_API_KEY`` is
available before the extractor/answerer construct. Real environment variables always
win — the file only fills in what isn't already set — and a missing python-dotenv is
non-fatal (then the key must come from the real environment).
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

from .config import Config
from .graph import KnowledgeGraph
from .rag import RagAnswer, SearchHit, SearchResult

#: Which line of development this code is, carried IN THE CODE rather than in package
#: metadata. The paired app shows it in Settings so you can tell at a glance which engine is
#: loaded (see BRANCH-NOTES.md). Metadata cannot answer that: an editable dev install keeps
#: whatever version it was first installed with (0.1.0 here), so `importlib.metadata` reports
#: a stale number while the paired code is what actually runs. A module constant travels with
#: the source in every install mode. Absent on main — readers must use getattr(..., None).
BRANCH = "staging"

__all__ = ["BRANCH", "Config", "KnowledgeGraph", "RagAnswer", "SearchHit", "SearchResult"]

# The single source of truth for the version is the installed package metadata,
# stamped into the wheel by the release pipeline (never hardcode a number here —
# a hardcoded string goes stale the moment the pipeline ships the next wheel).
# A source checkout that was never pip-installed has no metadata; report a dev
# placeholder rather than crash.
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("you-kg")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
