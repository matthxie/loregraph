"""Per-instance ingest-store cache for the LongMemEval per-instance benchmark harness.

Extraction is ~93% of a benchmark run's cost (kg/testrun.py run_per_instance). When a run
only changes QUERY-side code (retrieval/rerank/context/reader/judge), the store that
`g.ingest(sessions)` produces for a given instance is identical to a prior run's — so
re-running extraction is pure waste. This module hashes the ingest-relevant inputs
(instance id, session content, ingest-relevant config, extractor prompt text) into a
cache key, and copies a saved store on a hit instead of re-ingesting.

The cache is content-addressed and never auto-evicted; stale entries (e.g. after a
`local_backend` model upgrade with unchanged config field values) must be cleared
manually — `python -m kg cache-clear` or `rm -rf store/cache`.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3

from .config import Config
from .corpus import CorpusItem

# Config fields that change what `Ingestor.ingest` WRITES (extraction, canonicalization,
# chunking, derived-edge thresholds). Deliberately excludes: pure query-side fields
# (retrieval, rerank, context, reader, judge, PPR self-guard), the testrun-only completeness
# audit (completeness_tier2*, never used for extraction/canon/RAG), display-only back-compat
# fields (`extractor`, `embedder` — get_extractor/get_embedder actually key off
# extractor_backend/embed_model), and knobs that don't change written content
# (semaphore_limit is concurrency only, ingest_flush_every is a checkpoint cadence,
# community_seed only affects build_communities which always runs fresh after a cache hit —
# communities are never cached).
INGEST_RELEVANT_FIELDS = (
    "llm_model", "extractor_backend", "local_backend", "cue_escalate",
    "gliner_model", "gliner_threshold",
    "gliner2_model", "gliner2_entity_threshold", "gliner2_relation_threshold",
    "embed_model", "embed_dim",
    "reflexion", "long_doc_chars", "extract_max_chars", "extract_max_tokens", "lead_chars",
    "chunking", "chunk_target_chars", "chunk_max_chars", "part_of_weight", "next_weight",
    "syn_link_threshold", "syn_merge_threshold", "entropy_min_chars", "entropy_min_bits",
    "rel_syn_merge_threshold", "max_relation_labels",
    "l3_enabled", "l3_model", "rel_gray_floor",
    "episode_knn_k", "episode_knn_floor", "shared_min_overlap", "shared_edges", "shared_hub_cap",
    "self_entity", "self_name",
)


def _extractor_prompt_digest() -> str:
    """Hash the extractor system-prompt text, the emit_graph tool schema, AND the
    cue-gating regex source (kg/cues.py) so editing the prompt, changing the schema (e.g.
    adding a typed field), or changing which text earns an escalation call all
    auto-invalidate the cache even though no Config field changed."""
    import inspect
    import json

    from . import cues
    from .extractors import GRAPH_TOOL, OpenAIExtractor
    h = hashlib.sha256()
    h.update(OpenAIExtractor._SYS.encode("utf-8"))
    h.update(OpenAIExtractor._FIRST_PERSON_CLAUSE.encode("utf-8"))
    h.update(json.dumps(GRAPH_TOOL, sort_keys=True).encode("utf-8"))
    h.update(inspect.getsource(cues).encode("utf-8"))
    return h.hexdigest()


def _config_digest(config: Config) -> str:
    h = hashlib.sha256()
    for field in INGEST_RELEVANT_FIELDS:
        h.update(field.encode("utf-8"))
        h.update(b"=")
        h.update(repr(getattr(config, field)).encode("utf-8"))
        h.update(b";")
    return h.hexdigest()


def _sessions_digest(sessions: list[CorpusItem]) -> str:
    h = hashlib.sha256()
    for item in sessions:                       # loader yields them in chronological order
        content = item.text if item.modality == "text" else (item.image_path or "")
        h.update(item.id.encode("utf-8", "ignore"))
        h.update(b"\0")
        h.update((item.created_at or "").encode("utf-8"))
        h.update(b"\0")
        h.update((content or "").encode("utf-8", "ignore"))
        h.update(b"\0")
    return h.hexdigest()


def ingest_cache_key(instance_id: str, sessions: list[CorpusItem], config: Config) -> str:
    """Stable hex digest over (instance id, session content, ingest-relevant config slice,
    extractor prompt text). A queryside-only config change must NOT change this key."""
    h = hashlib.sha256()
    h.update(instance_id.encode("utf-8", "ignore"))
    h.update(_sessions_digest(sessions).encode("utf-8"))
    h.update(_config_digest(config).encode("utf-8"))
    h.update(_extractor_prompt_digest().encode("utf-8"))
    return h.hexdigest()


def cache_dir_for(store_path: str) -> str:
    return os.path.join(os.path.dirname(store_path) or "store", "cache")


def cache_path(store_path: str, instance_id: str, key: str) -> str:
    return os.path.join(cache_dir_for(store_path), f"{instance_id}-{key[:12]}.db")


def _sqlite_copy(src_path: str, dst_path: str) -> None:
    """Copy one SQLite db to another via the backup API — safe regardless of the source's
    WAL checkpoint state (unlike a raw file copy, which can miss uncommitted WAL pages)."""
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        p = dst_path + suffix
        if os.path.exists(p):
            os.remove(p)
    src_con = sqlite3.connect(src_path)
    dst_con = sqlite3.connect(dst_path)
    try:
        src_con.backup(dst_con)
    finally:
        dst_con.close()
        src_con.close()


def try_restore(store_path: str, instance_id: str, key: str) -> bool:
    """On a hit, copy the cached db to `store_path` and return True — the working store is
    a copy, never the cache file itself, so a query-side bug can never corrupt the cache."""
    src = cache_path(store_path, instance_id, key)
    if not os.path.exists(src):
        return False
    _sqlite_copy(src, store_path)
    return True


def save(store_path: str, instance_id: str, key: str) -> None:
    """After a fresh ingest, copy the resulting store into the cache for future runs."""
    _sqlite_copy(store_path, cache_path(store_path, instance_id, key))
