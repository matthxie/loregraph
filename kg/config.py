"""Central configuration — every tunable from the design lives here.

The thresholds map to docs/ARCHITECTURE.md §3 (drift control), §4 (embeddings) and §5
(retrieval). They are deliberately conservative / link-biased (under-merge) per §9 risk 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- models (ARCHITECTURE §0) -------------------------------------------
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384  # bge-small / MiniLM both 384; hashing fallback matches

    # ---- backend selection (LIVE-ONLY; the offline heuristic/hashing backends were
    #      removed). Kept as fields for back-compat: extractor→Haiku, embedder→bge. ----
    extractor: str = "haiku"
    embedder: str = "st"

    # ---- extractor backend ---------------------------------------------------
    # DEFAULT "cue_gated" = the production strategy: a free local NLP floor
    # (`local_backend`, gliner_yake_cooccur) on every entry, plus ONE Haiku call
    # ONLY on entries carrying a termination/relative-date/identity cue (kg/cues.py).
    # Other choices: "haiku" (full LLM on everything), or a pure LLM-free / hybrid NLP
    # backend (gliner_yake_cooccur | gliner2 | keyword_only | … , see kg/nlp_extractors.py).
    extractor_backend: str = "cue_gated"
    local_backend: str = "gliner_yake_cooccur"   # the always-on local floor under cue_gated
    cue_escalate: bool = True                     # call Haiku on cue-bearing entries (needs key)
    gliner_model: str = "urchade/gliner_small-v2.1"
    gliner_threshold: float = 0.5
    # GLiNER2 (fastino) — ONE local encoder that emits typed entities AND typed relations, run in
    # batched forward passes per section (kg/nlp_extractors.py Gliner2Extractor). $0, on-device.
    gliner2_model: str = "fastino/gliner2-large-v1"
    gliner2_entity_threshold: float = 0.5    # higher trims generic-concept over-extraction
    gliner2_relation_threshold: float = 0.5  # higher = cleaner, fewer spurious edges

    # ---- ingestion (§6) ------------------------------------------------------
    semaphore_limit: int = 5          # bounded LLM concurrency (graphiti SEMAPHORE_LIMIT)
    reflexion: bool = True            # one extra recall pass after extraction
    long_doc_chars: int = 6000        # above this, extract section-by-section
    # Per-call input cap inside Extractor.extract_text. MUST be >= long_doc_chars so a
    # window-sized section is never silently truncated (a section is exactly long_doc_chars
    # wide). Swept together with long_doc_chars in the lever-2 window A/B (optimization.md).
    extract_max_chars: int = 12000
    # Output ceiling (max_tokens) on the emit_graph tool call. The model emits the WHOLE
    # entity/tag/relation payload under this cap; a section whose graph exceeds it is silently
    # truncated mid-JSON (tail dropped — content already generated and paid for). Raised
    # 1500→4000 after the max_tokens A/B (2026-06-25, optimization.md lever 2): the old 1500
    # truncated ~8/246 calls on the shipped 6000 window; 4000 clears all truncation and is the
    # knee (richness plateaus — mt8000 adds nothing). Billed only on emitted tokens, so ~+0.2%.
    extract_max_tokens: int = 4000
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
    l3_model: str = "gpt-4o-mini"
    rel_gray_floor: float = 0.90

    # ---- derived edges (§2 / §6.5) ------------------------------------------
    episode_knn_k: int = 6            # SIMILAR_TO neighbours per episode
    episode_knn_floor: float = 0.55   # min cosine for an episode↔episode SIMILAR_TO edge
    shared_min_overlap: int = 1       # min shared tags/entities for a SHARED_* edge
    # SHARED_TAG / SHARED_ENTITY derivation. Derivation is incremental (only pairs that
    # involve a NEW episode are computed), but a popular hub still pairs each new episode
    # against every member — shared_hub_cap bounds that at the `cap` most recent members
    # (a hub big enough to hit the cap has high df and near-zero IDF weight anyway).
    # shared_edges=False skips SHARED_* entirely — the removal A/B: the tag/entity star
    # already connects the same episodes at path length 2, so PPR may not need shortcuts.
    shared_edges: bool = True
    shared_hub_cap: int = 256

    # ---- write-through flush cadence ----------------------------------------
    # The ingest loop checkpoints the store to SQLite every N episodes (and always at the
    # end), so a crash mid-ingest loses at most one window. 0 disables mid-loop flushes.
    ingest_flush_every: int = 200

    # ---- retrieval (§5) ------------------------------------------------------
    ppr_damping: float = 0.5          # HippoRAG personalization damping
    seed_k: int = 10                  # seed nodes from fused embedding+BM25
    top_k: int = 8                    # episodes returned to the caller
    mmr_lambda: float = 0.6           # MMR relevance↔diversity tradeoff
    inferred_confidence_floor: float = 0.3  # drop INFERRED edges below this in traversal

    # ---- answer-path pipeline (the production read strategy; used by `ask`) ----
    # On top of the PPR pool the answer path adds: a 4-lane query router (kg/route.py),
    # a fact-bearing-episode augment on state/evolution lanes, and a cross-encoder rerank
    # of the candidate pool — but ONLY on the hard lanes (rerank_lanes), since the
    # cross-encoder can demote the gold on easy single-fact lookups. All default ON.
    route: bool = True                # 4-lane query router (recency|state|multihop|single)
    rerank: bool = True               # cross-encoder rerank of the candidate pool
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 32             # candidates fed to the cross-encoder
    rerank_lanes: tuple = ("state", "multihop")   # lanes that get reranked
    fact_lane_augment: bool = True    # surface fact-bearing episodes on state/evolution lanes

    # ---- RAG answer flow (§5) — PPR builds the context, the LLM does NOT traverse ----
    # The query path is retrieve-then-read: PPR (or as-of-T PPR) assembles a context blob
    # of the top episodes + the currently-valid facts among the touched entities, then a
    # SINGLE LLM call answers over it with citations. No per-hop LLM walking. LIVE-ONLY:
    # the offline answerer was removed, so `kg ask` requires a key (or an injected client).
    rag_backend: str = "openai"        # "openai" | "auto" (both = live OpenAI)
    rag_model: str = "gpt-4o-mini"
    rag_max_tokens: int = 1024        # answer output cap
    rag_context_episodes: int = 8     # episodes whose text enters the context blob (== top_k,
                                       # so nothing recall@k=8 finds gets truncated before the
                                       # reader sees it; was 6 — see optimization.md baseline)
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

    # ---- self-anchor PPR hub guard (only bites when self_entity is on) -------
    # In personal-web mode every first-person reference resolves to ONE node
    # (SELF_ENTITY_ID), so on a deep single-user graph that node becomes incident
    # to a large fraction of RELATED_TO fact edges — a high-degree hub. idf_weight
    # down-weights it as a SEED but does NOT stop PageRank routing activation
    # THROUGH it during the diffusion, over-spreading mass to weakly-related
    # episodes. This guard controls how the self node participates in the PPR
    # projection. "none" = byte-for-byte unchanged (default, safe):
    #   none     — no guard; self diffuses like any other entity
    #   exclude  — drop self + its RESOLVES_TO star from the projection entirely
    #              (like IN_COMMUNITY); self carries no discriminating signal
    #   cap      — keep self but cap its incident edge weights (self_guard_cap)
    #   seed     — keep self in the graph but never SEED it (no personalization mass)
    self_guard: str = "none"          # none | exclude | cap | seed
    self_guard_cap: float = 0.05      # 'cap' mode: max weight per self-incident edge

    # ---- misc ----------------------------------------------------------------
    random_seed: int = 42
    extra: dict = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Config":
        return cls()
