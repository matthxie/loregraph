"""Central configuration — every tunable lives here, in three tiers.

Tier 1 = the primary knobs most users tweak. Tier 2 = behavior toggles whose defaults
are the banked A/B winners (each "off" value reproduces pre-feature behavior). Tier 3 =
experimental thresholds/caps tuned against the eval sets — don't touch unless you know
why. Rationale and A/B provenance for every field: docs/CONFIG.md. Architecture context:
docs/ARCHITECTURE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ══════════ TIER 1 — primary ══════════════════════════════════════════════
    llm_model: str | None = None                  # extraction LLM
    rag_model: str | None = None                  # answerer used by `kg ask`
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384                          # must match embed_model's output dim
    extractor_backend: str = "auto"               # "auto" = LLM on every entry when a signed-in
    #                                               provider is live (kg/llm_client.py), else the
    #                                               keyless cue_gated floor; also
    #                                               "cue_gated"|"llm"|NLP backends (nlp_extractors.py)
    chunking: str = "turns"                       # "none"|"turns"|"markdown"|"prose"|"code"|"auto"
    top_k: int = 8                                # episodes returned to the caller
    self_entity: bool = False                     # personal-web first-person resolution
    self_name: str = "self"                       # display name of the self anchor

    # ══════════ TIER 2 — behavior toggles (defaults = banked winners) ═════════
    reflexion: bool = True            # one extra recall pass after extraction
    cue_escalate: bool = True         # cue_gated: LLM call on cue-bearing entries (needs key)
    shared_edges: bool = True         # derive SHARED_TAG / SHARED_ENTITY shortcut edges
    route: bool = True                # 4-lane query router (recency|state|multihop|single)
    rerank: bool = True               # cross-encoder rerank of the candidate pool
    fact_lane_augment: bool = True    # surface fact-bearing episodes on state/evolution lanes
    rag_provenance_promote: bool = True   # pull a fact's source chunk into context
    rag_answer_events: str = "lanes"  # structured event enumeration: "off"|"lanes"|"all"
    rag_answer_events_lanes: tuple = ("multihop", "state")
    agg_reconcile: bool = False       # answer-time aggregation reconciliation: recompute the
    #                                   aggregate from the reader's enumerated events[] and, on
    #                                   a mismatch with the number stated in the answer text,
    #                                   append a correction citing the enumeration. QUERY-SIDE;
    #                                   off = answer flow byte-identical (docs/OFFLINE_EVAL.md
    #                                   Round 6a). NOT an ingest-cache field.
    agg_map_reduce: bool = False      # map-reduce aggregation lane: on MULTIHOP + aggregate-
    #                                   shaped questions, one MAP LLM call per source session
    #                                   enumerates matching items, CODE merges/dedups/counts
    #                                   or sums, and a REDUCE call answers over the original
    #                                   context PLUS the computed table. QUERY-SIDE; off =
    #                                   byte-identical. NOT an ingest-cache field.
    facts_projection: bool = False    # on flush/save, (re)generate two derived SQLite
    #                                   tables (facts_view, agg_view) in the same db file,
    #                                   rebuilt WHOLESALE from the RELATED_TO edges every
    #                                   flush — a SQL-queryable projection of the graph's
    #                                   fact/occurrence data. The LOAD path never reads them
    #                                   (delete + reload = identical graph). QUERY-SIDE; off =
    #                                   save() byte-identical, no extra tables. NOT an
    #                                   ingest-cache field (docs/OFFLINE_EVAL.md Round 6b).
    agg_evidence: bool = False        # on aggregate-shaped questions (is_aggregate_question),
    #                                   append a "GRAPH TALLIES (may be incomplete…)" section
    #                                   to the context: per-pair occurrence tallies for the
    #                                   anchor entities, computed IN-MEMORY from believed
    #                                   RELATED_TO edges (works without facts_projection and on
    #                                   read-only stores). EVIDENCE, not an oracle — capped ~10
    #                                   lines, caveat header. QUERY-SIDE; off = context
    #                                   byte-identical (docs/OFFLINE_EVAL.md Round 6b).
    fact_vectors: bool = False        # embed a statement-granularity surface per believed
    #                                   RELATED_TO edge ("<src> <rel> <dst>") + a distilled
    #                                   frequency surface per recurring (src,rel,dst) group,
    #                                   stored under vector kind="fact" — makes each fact a
    #                                   first-class retrieval target (docs/OFFLINE_EVAL.md
    #                                   Round 7a, sharp edge #6). Local embed ($0). AFFECTS
    #                                   WHAT INGEST WRITES → it IS an ingest-cache field
    #                                   (hashed only when ON; existing off caches stay valid,
    #                                   backfilled via `kg backfill-fact-vectors`).
    fact_lane: bool = False           # statement-granularity retrieval lane: score the query
    #                                   against the kind="fact" vectors (config.fact_vectors /
    #                                   `kg backfill-fact-vectors`), map the top fact_lane_k
    #                                   hits back to their PROVENANCE episodes + endpoint
    #                                   entities, and MERGE those as seed candidates ADDITIVELY
    #                                   — never displacing or down-weighting an episode-lane
    #                                   seed, total fact-lane mass capped at fact_lane_weight ×
    #                                   the episode-lane mass. Gets a fact's asserting chunk into
    #                                   the PPR pool because its CLAIM matched, and marks the
    #                                   matched lines in the FACTS section (docs/OFFLINE_EVAL.md
    #                                   Round 7b). QUERY-SIDE; off = seeds/pool/context
    #                                   byte-identical. NOT an ingest-cache field (reads the
    #                                   vectors fact_vectors wrote; no vectors → lane no-ops).
    fact_lane_k: int = 10             # top-N fact/aggregate vector hits the lane seeds from
    fact_lane_weight: float = 0.5     # cap: Σ fact-lane seed mass ≤ this × Σ episode-lane mass
    history_all_lanes: bool = False   # serve the closed-fact HISTORY delta on EVERY lane,
    #                                   not just STATE (offline eval variant A, amended:
    #                                   outside STATE only CLOSED lines render — open lines
    #                                   duplicate the FACTS section ~90%)
    event_facts: bool = False         # event-shaped predicates (went_to/visited/attended…)
    #                                   write CLOSED [d,d] occurrence edges + event=True
    #                                   instead of open [d,∞) states (docs/PIPELINE.md
    #                                   sharp edge #1); off = pre-fix write semantics
    ingest_date_filter: bool = False  # apply the deterministic date/numeric term filter
    #                                   (_filter_date_terms) to EVERY extraction at ingest,
    #                                   regardless of backend — catches date-endpoint junk
    #                                   from the local NLP floor too. Default off: hashed
    #                                   into the ingest-cache key only when ON, so all
    #                                   existing cached stores stay valid until the next
    #                                   paid re-ingest flips it on.
    date_window_boost: bool = False   # reserve slots for episodes in a resolved date window
    rag_session_dedup: bool = False   # distinct sessions in the context-eligible prefix
    rag_resolve_reldates: bool = False    # annotate relative dates in episode text at render
    rag_backend: str = "openai"       # answerer backend: "openai" | "auto"
    judge_model: str = "gpt-4o"       # LLM-as-judge scoring only; never touches prod cost
    completeness_tier2: bool = True   # LLM occurrence audit (testrun/dashboard only)
    completeness_tier2_model: str = "gpt-4o-mini"
    ingest_flush_every: int = 200     # checkpoint store every N episodes; 0 = end only

    # ══════════ TIER 3 — experimental / internals (see docs/CONFIG.md) ════════
    # -- back-compat (live-only backends; values other than these are gone) ----
    extractor: str = "llm"
    embedder: str = "st"

    # -- local NLP extraction stack --------------------------------------------
    local_backend: str = "gliner_yake_cooccur"    # the always-on floor under cue_gated
    gliner_model: str = "urchade/gliner_small-v2.1"
    gliner_threshold: float = 0.5
    gliner2_model: str = "fastino/gliner2-large-v1"
    gliner2_entity_threshold: float = 0.5
    gliner2_relation_threshold: float = 0.5

    # -- ingestion windows / caps ----------------------------------------------
    semaphore_limit: int = 40         # bounded LLM concurrency
    long_doc_chars: int = 6000        # above this, extract section-by-section
    extract_max_chars: int = 12000    # per-call input cap; MUST be >= long_doc_chars
    extract_max_tokens: int = 4000    # emit_graph output ceiling; too low truncates mid-JSON
    lead_chars: int = 2000            # embed the lead section for very long docs

    # -- tag drift control: L1/L2 merge thresholds -----------------------------
    syn_link_threshold: float = 0.85  # ENTITY cosine > → SIMILAR_TO link (don't merge)
    syn_merge_threshold: float = 0.93 # ENTITY cosine > → candidate hard merge
    # Tags get their OWN, looser thresholds so near-synonym tags actually link/merge
    # ("check if similar; if not, make own") WITHOUT endangering entity resolution —
    # entities keep the high bars above, tags use these. Decoupling is the fix for the
    # documented threshold collision (entities want high, tags want low).
    tag_syn_link_threshold: float = 0.80   # TAG cosine > → SIMILAR_TO link (connect)
    tag_syn_merge_threshold: float = 0.88  # TAG cosine > → hard merge (collapse dup)
    entropy_min_chars: int = 4        # shorter tags merge on exact match only
    entropy_min_bits: float = 2.0     # min Shannon entropy to allow fuzzy merge
    rel_syn_merge_threshold: float = 0.95
    max_relation_labels: int = 3      # cap labels carried by one connection

    # -- confidence-gated closure (docs/TEMPORAL.md; kg/temporal.py) -----------
    # A close / supersede / retract fires only when the asserting fact is at least about as
    # trustworthy as the fact it would overturn. If the INCOMING confidence is more than
    # `dispute_confidence_margin` BELOW the stored edge's, we do NOT close it — a low-trust
    # claim can't silently kill a high-trust fact. Instead the stored edge records the losing
    # claim in `disputed_by` (episode + confidence) so the disagreement is visible and the
    # user can adjudicate. Set the margin to 1.0 to disable (nothing is ever gated) — the
    # pre-gate behaviour.
    dispute_confidence_margin: float = 0.3

    # -- L3 selective LLM canonicalization tie-breaker (ship-disabled) ---------
    l3_enabled: bool = False
    l3_model: str | None = None       # None = provider default (see llm_model note above)
    rel_gray_floor: float = 0.90      # gray zone lower bound fed to the L3 adjudicator

    # -- derived edges ----------------------------------------------------------
    episode_knn_k: int = 6            # SIMILAR_TO neighbours per episode
    episode_knn_floor: float = 0.55   # min cosine for an episode↔episode edge
    shared_min_overlap: int = 1       # min shared tags/entities for a SHARED_* edge
    shared_hub_cap: int = 256         # cap pairing against a hub's most recent members

    # -- chunking geometry ------------------------------------------------------
    chunk_target_chars: int = 2200    # greedy-pack natural units up to ~this per chunk
    chunk_max_chars: int = 4400       # unchunked threshold / hard ceiling per packed unit
    part_of_weight: float = 0.3       # PART_OF traversal weight (parent ≠ super-hub)
    next_weight: float = 0.5          # NEXT (sequence) traversal weight

    # -- retrieval: PPR ---------------------------------------------------------
    ppr_damping: float = 0.5          # HippoRAG personalization damping
    ppr_backend: str = "global"       # "global" (exact) | "push" (local, eps-approximate)
    ppr_push_eps: float = 1e-6
    seed_k: int = 10                  # seed nodes from fused embedding+BM25
    mmr_lambda: float = 1.0           # MMR relevance↔diversity (1.0 = pure relevance)
    inferred_confidence_floor: float = 0.3  # drop INFERRED edges below this in traversal

    # -- answer-path pipeline ---------------------------------------------------
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 32             # candidates fed to the cross-encoder
    rerank_lanes: tuple = ("state", "multihop")   # only hard lanes get reranked
    rerank_keep_ppr_top: int = 3      # raw-PPR top-N always kept in final top-k; 0 = off
    seed_reserve: int = 0             # reserved seats for seed-ranked sessions; 0 = off
    date_window_slots: int = 2        # slots reserved under date_window_boost

    # -- RAG context assembly ---------------------------------------------------
    rag_max_tokens: int = 4096        # answer output cap
    rag_context_episodes: int = 5     # episodes whose text enters the context blob
    rag_episode_chars: int = 20000    # per-episode text budget in the context
    rag_max_facts: int = 30           # currently-valid facts surfaced in the context
    rag_chunks_per_source: int = 2    # context slots one source may occupy; 0 = off
    rag_parent_expand: int = 2        # sibling-chunk expansion radius; 0 = off
    rag_expand_budget_chars: int = 60000  # stop adding siblings past this total
    rag_retarget: str = "ce"          # chunk retargeting: "off"|"seed"|"seed+lex"|"ce"|"ce+seed"

    # -- self-anchor PPR hub guard (only bites when self_entity is on) ---------
    self_guard: str = "none"          # none | exclude | cap | seed
    self_guard_cap: float = 0.05      # 'cap' mode: max weight per self-incident edge

    # -- misc -------------------------------------------------------------------
    community_seed: int = 42          # pin for reproducible community ids
    directed: bool = True             # graph direction
    random_seed: int = 42
    extra: dict = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Config":
        return cls()
