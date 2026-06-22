"""Central configuration — every tunable from the design lives here.

The thresholds map directly to docs/ARCHITECTURE.md §3 (drift control), §4
(embeddings) and §5 (retrieval). They are deliberately conservative / link-biased
(under-merge) per §9 risk 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- models (ARCHITECTURE §0) -------------------------------------------
    llm_model: str = "claude-haiku-4-5-20251001"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384  # bge-small / MiniLM both 384; hashing fallback matches

    # ---- backend selection: "auto" | "haiku"/"heuristic" | "st"/"hashing" ----
    extractor: str = "auto"
    embedder: str = "auto"

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
    # Predicate labels are consolidated like topical tags, but with a HIGHER merge
    # bar: antonyms/inverses ("is_friend_of" vs "is_enemy_of", "manages" vs
    # "managed_by") sit close in embedding space, so we merge conservatively and
    # never auto-link across them.
    rel_syn_merge_threshold: float = 0.95
    max_relation_labels: int = 3      # cap labels carried by one connection

    # ---- L3 selective LLM canonicalization tie-breaker (§3 L3 — SHIP DISABLED) ----
    # Fires ONLY on the residual ambiguous band the deterministic L1/L2 tiers can't
    # confidently resolve, and (for relations) ONLY behind the deterministic antonym/
    # inverse/passive veto in canonicalize.py. Decided INSIDE resolve_* before a node is
    # minted (sequential, main-thread), so a MERGE just returns the existing id — no
    # provisional node, no edge rewrite. Default OFF and auto-skipped with no
    # ANTHROPIC_API_KEY, so the offline path is byte-for-byte unchanged. Enable only
    # after `python -m kg eval-canon` shows zero antonym/inverse false-merges. See CLAUDE.md.
    l3_enabled: bool = False
    l3_model: str = "claude-haiku-4-5-20251001"  # pinned adjudicator model (temperature=0)
    rel_gray_floor: float = 0.90      # relation gray band = [rel_gray_floor, rel_syn_merge_threshold)
    # entity/tag gray band reuses [syn_link_threshold, syn_merge_threshold) = [0.85, 0.93)

    # ---- derived edges (§2 / §6.5) ------------------------------------------
    object_knn_k: int = 6             # SIMILAR_TO neighbours per object
    object_knn_floor: float = 0.55    # min cosine for an object↔object SIMILAR_TO edge
    shared_min_overlap: int = 1       # min shared tags/entities for a SHARED_* edge

    # ---- retrieval (§5) ------------------------------------------------------
    ppr_damping: float = 0.5          # HippoRAG personalization damping
    seed_k: int = 10                  # seed nodes from fused embedding+BM25
    top_k: int = 8                    # objects returned to the caller
    mmr_lambda: float = 0.6           # MMR relevance↔diversity tradeoff
    inferred_confidence_floor: float = 0.3  # drop INFERRED edges below this in traversal

    # ---- communities (§ phase 3) --------------------------------------------
    community_seed: int = 42          # pin for reproducible community ids

    # ---- graph direction (§2 rev 3) -----------------------------------------
    # Edges are stored DIRECTED (MultiDiGraph) so relationship semantics survive
    # (src→dst). Traversal/diffusion (PPR, BFS) runs over a SYMMETRIZED projection
    # so recall is unchanged (HippoRAG runs PPR undirected); see retrieval.py.
    directed: bool = True

    # ---- misc ----------------------------------------------------------------
    random_seed: int = 42
    extra: dict = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Config":
        return cls()
