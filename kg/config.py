"""Central configuration — every tunable from the design lives here.

The thresholds map to docs/ARCHITECTURE.md §3 (drift control), §4 (embeddings) and §5
(retrieval). They are deliberately conservative / link-biased (under-merge) per §9 risk 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- models (ARCHITECTURE §0) -------------------------------------------
    llm_model: str = "claude-haiku-4-5-20251001"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384  # bge-small / MiniLM both 384; hashing fallback matches

    # ---- backend selection (LIVE-ONLY; the offline heuristic/hashing backends were
    #      removed). Kept as fields for back-compat: extractor→Haiku, embedder→bge. ----
    extractor: str = "haiku"
    embedder: str = "st"

    # ---- ingestion (§6) ------------------------------------------------------
    semaphore_limit: int = 5          # bounded LLM concurrency (graphiti SEMAPHORE_LIMIT)
    reflexion: bool = True            # one extra recall pass after extraction
    long_doc_chars: int = 6000        # above this, extract section-by-section
    lead_chars: int = 2000            # embed the lead section for very long docs

    # ---- tag drift control (§3) ---------------------------------------------
    syn_link_threshold: float = 0.85  # cosine > → SIMILAR_TO link (don't merge)
    syn_merge_threshold: float = 0.93 # cosine > → candidate hard merge
    entropy_min_chars: int = 4        # entropy guard: shorter tags merge on exact only
    entropy_min_bits: float = 2.0     # min Shannon entropy (chars) to allow fuzzy merge

    # ---- relationship-tag consolidation (§3b — open-vocab relation canonicalization)
    rel_syn_merge_threshold: float = 0.95
    max_relation_labels: int = 3      # cap labels carried by one connection

    # ---- L3 selective LLM canonicalization tie-breaker (§3 L3 — SHIP DISABLED) ----
    l3_enabled: bool = False
    l3_model: str = "claude-haiku-4-5-20251001"
    rel_gray_floor: float = 0.90

    # ---- derived edges (§2 / §6.5) ------------------------------------------
    episode_knn_k: int = 6            # SIMILAR_TO neighbours per episode
    episode_knn_floor: float = 0.55   # min cosine for an episode↔episode SIMILAR_TO edge
    shared_min_overlap: int = 1       # min shared tags/entities for a SHARED_* edge

    # ---- retrieval (§5) ------------------------------------------------------
    ppr_damping: float = 0.5          # HippoRAG personalization damping
    seed_k: int = 10                  # seed nodes from fused embedding+BM25
    top_k: int = 8                    # episodes returned to the caller
    mmr_lambda: float = 0.6           # MMR relevance↔diversity tradeoff
    inferred_confidence_floor: float = 0.3  # drop INFERRED edges below this in traversal

    # ---- RAG answer flow (§5) — PPR builds the context, the LLM does NOT traverse ----
    # The query path is retrieve-then-read: PPR (or as-of-T PPR) assembles a context blob
    # of the top episodes + the currently-valid facts among the touched entities, then a
    # SINGLE LLM call answers over it with citations. No per-hop LLM walking. LIVE-ONLY:
    # the offline answerer was removed, so `kg ask` requires a key (or an injected client).
    rag_backend: str = "claude"       # "claude" | "auto" (both = live Claude)
    rag_model: str = "claude-haiku-4-5-20251001"
    rag_max_tokens: int = 1024        # answer output cap
    rag_context_episodes: int = 6     # episodes whose text enters the context blob
    rag_episode_chars: int = 1200     # per-episode text budget in the context
    rag_max_facts: int = 30           # currently-valid facts surfaced in the context

    # ---- communities (§ phase 3) --------------------------------------------
    community_seed: int = 42          # pin for reproducible community ids

    # ---- graph direction (§2) -----------------------------------------------
    directed: bool = True

    # ---- personal-web mode (optional) ---------------------------------------
    # When on, first-person references ("i"/"me"/"my"/…) resolve to ONE stable
    # canonical "self" anchor so the narrator's relationships form on a single node.
    # OFF by default — the offline path is byte-for-byte unchanged when off.
    self_entity: bool = False     # personal-web first-person resolution
    self_name: str = "self"       # display name of the self anchor

    # ---- misc ----------------------------------------------------------------
    random_seed: int = 42
    extra: dict = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Config":
        return cls()
