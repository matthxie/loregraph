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
    semaphore_limit: int = 40         # bounded LLM concurrency (graphiti SEMAPHORE_LIMIT)
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

    # ---- LLM-as-judge grading model (testrun/ablate/guard_eval scoring only —
    # never used for extraction/canon/RAG, so bumping this doesn't touch prod cost) ----
    judge_model: str = "gpt-4o"

    # ---- extraction-completeness audit (testrun/dashboard only — see kg/completeness.py
    #      and spikes/completeness/REPORT.md; never used for extraction/canon/RAG) ----
    completeness_tier2: bool = True   # LLM occurrence audit; tier 1 (regex) is always on
    completeness_tier2_model: str = "gpt-4o-mini"

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

    # ---- episode chunking (natural boundaries; kg/chunkers.py) ---------------
    # "turns" splits each text entry along its natural structure (chat turns, else
    # paragraphs; oversized units fall back to sentence packing) into several small
    # episodes, keeps the full original on an un-rankable SOURCE parent, and links
    # chunk→parent (PART_OF) + chunk→next-sibling (NEXT). Retrieval then ranks
    # statement-grained chunks instead of whole multi-thousand-char blobs. "none" =
    # legacy one-episode-per-entry (byte-for-byte unchanged). Phase 2 adds explicit
    # "markdown" (heading sections + breadcrumb prefixes) / "prose" (paragraph→sentence
    # packing) / "code" (top-level blank-line blocks) chunkers plus "auto", which
    # sniffs the format PER ENTRY (kg/chunkers.py sniff_format) and routes to the
    # right chunker — chat text still gets exactly the "turns" behavior.
    chunking: str = "none"            # "none"|"turns"|"markdown"|"prose"|"code"|"auto"
    chunk_target_chars: int = 2200    # greedy-pack natural units up to ~this per chunk
    chunk_max_chars: int = 4400      # entries at/below this stay unchunked; also the
                                      # hard ceiling one packed unit may reach
    part_of_weight: float = 0.3       # PART_OF traversal weight (parent must not
    next_weight: float = 0.5          # become a sibling super-hub); NEXT = sequence

    # ---- write-through flush cadence ----------------------------------------
    # The ingest loop checkpoints the store to SQLite every N episodes (and always at the
    # end), so a crash mid-ingest loses at most one window. 0 disables mid-loop flushes.
    ingest_flush_every: int = 200

    # ---- retrieval (§5) ------------------------------------------------------
    ppr_damping: float = 0.5          # HippoRAG personalization damping
    # PPR backend: "global" = exact power iteration over the whole projection (O(N+E)
    # per query); "push" = local forward push (Andersen–Chung–Lang), explores only the
    # seed neighborhood — work independent of graph size, eps-approximate scores
    # (ppr_push_eps, per unit of weighted degree). Same fixed point, same ranking at
    # retrieval resolution; "global" (default) is byte-identical to today.
    ppr_backend: str = "global"       # "global" | "push"
    ppr_push_eps: float = 1e-6
    seed_k: int = 10                  # seed nodes from fused embedding+BM25
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
    # PPR guarantee under the cross-encoder: the top-N episodes of the RAW PPR pool are
    # always kept in the final top-k even if the reranker pushes them below k (the CE can
    # demote a gold the graph ranked #1 completely out of the context). 0 disables.
    rerank_keep_ppr_top: int = 3
    fact_lane_augment: bool = True    # surface fact-bearing episodes on state/evolution lanes
    # Seed-rank reserved seats: up to N final top-k slots go to the highest raw-seed-score
    # episodes (embedding+BM25, the Seeder's own ranking) whose SESSION won no slot from
    # the PPR->MMR->CE pipeline — the containment eval showed 13/14 missing gold sessions
    # were already in seeds and were ranked out downstream. Displaces only the tail of the
    # final ranking; 0 (default) disables, byte-identical to today.
    seed_reserve: int = 0
    # Temporal date-window boost: resolve a relative-date phrase in the query ("last
    # Saturday", "two months ago") against as_of, then reserve up to date_window_slots
    # final slots for pool/seed episodes whose event date falls inside the resolved
    # window and whose session is not already represented. Off by default.
    date_window_boost: bool = False
    date_window_slots: int = 2
    # Session diversity in the context-eligible prefix: reorder the final ranking so the
    # first rag_context_episodes slots hold chunks from DISTINCT sessions (first chunk of
    # each new session in rank order; duplicates follow). With mmr_lambda=1.0 there is no
    # diversity pressure at all, so 2 of 5 reader slots routinely go to a second chunk of
    # a session already present while a gold session sits at rank 6. Off by default.
    rag_session_dedup: bool = False

    # ---- RAG answer flow (§5) — PPR builds the context, the LLM does NOT traverse ----
    # The query path is retrieve-then-read: PPR (or as-of-T PPR) assembles a context blob
    # of the top episodes + the currently-valid facts among the touched entities, then a
    # SINGLE LLM call answers over it with citations. No per-hop LLM walking. LIVE-ONLY:
    # the offline answerer was removed, so `kg ask` requires a key (or an injected client).
    rag_backend: str = "openai"        # "openai" | "auto" (both = live OpenAI)
    rag_model: str = "gpt-4o-mini"
    rag_max_tokens: int = 1024        # answer output cap
    rag_context_episodes: int = 5     # episodes whose text enters the context blob (== top_k,
                                       # so nothing recall@k=8 finds gets truncated before the
                                       # reader sees it; was 6 — see optimization.md baseline)
    rag_episode_chars: int = 20000    # per-episode text budget in the context
    rag_max_facts: int = 30           # currently-valid facts surfaced in the context
    # With chunking on, ranks 1..n can all be chunks of ONE source; cap how many context
    # slots a single source may occupy so the reader still sees other sessions. 0 = off.
    rag_chunks_per_source: int = 4
    # Sibling-chunk expansion (query-side only, context-only — see spikes/queryside/REPORT.md):
    # after top-n selection, pull in each selected chunk's #cNNN neighbours within this radius
    # so a chunked session's answer-bearing sibling isn't left out of context even when it
    # didn't rank into the top-n itself. 0 = off, byte-identical to pre-expansion behavior.
    rag_parent_expand: int = 0
    # Hard cap on total episode text after expansion; once hit, stop adding siblings (all
    # originally selected chunks are always kept — only expansion siblings are capped).
    rag_expand_budget_chars: int = 60000
    # Chunk-level retargeting (query-side only — see spikes/retarget/REPORT.md): the right
    # SOURCE can win a seat via _select_episodes while the wrong CHUNK of it gets picked
    # (PPR ranks by diffused chunk score, not by which chunk actually answers the question).
    # "off" (default) = byte-identical to pre-retargeting behavior.
    #   seed     — within each source that won seats, refill its slots with that source's
    #              best chunks by raw embedding seed rank (RetrievalResult.seed_scores)
    #              instead of PPR chunk order. Same slot count per source, swaps only.
    #   seed+lex — seed retarget, then a lexical-overlap swap pass: a same-source chunk NOT
    #              selected may swap in for a selected one if it strictly beats it on
    #              question content-word / digit-token overlap. The best-ranked (incumbent)
    #              chunk of each source is never swapped out.
    #   ce       — within each source that won seats, rank its chunks by local
    #              cross-encoder question<->chunk relevance (rerank_model, $0) instead of
    #              seed/PPR order; falls back to seed order if the model can't load.
    #   ce+seed  — blend: a source's slots split across the CE ranking and the seed
    #              ranking (best-rank-of-either), since the two signals miss on
    #              disjoint questions (CE: relevance != answer-bearing; seed: synonymy).
    rag_retarget: str = "off"           # "off" | "seed" | "seed+lex" | "ce" | "ce+seed"
    # Provenance promotion: a FACT's episode_id (the chunk it was extracted from) is pulled
    # into context if the fact's src/dst entity names overlap the question terms and that
    # chunk isn't already present — displacing only the lowest-ranked expansion sibling
    # (never an originally selected chunk). Runs after sibling expansion.
    rag_provenance_promote: bool = False
    # Structured enumeration in the answer tool (query-side only, reader-facing): on the
    # configured lanes the submit_answer schema REQUIRES an `events` array (date +
    # description + quantity) filled BEFORE the answer, turning "count in your head" into
    # "fill the list, then count" — the reader's dominant failure on aggregation questions
    # is stating a total without enumerating the events behind it. Same single LLM call,
    # a few dozen extra output tokens. "off" (default) = schema byte-identical to today.
    rag_answer_events: str = "off"     # "off" | "lanes" | "all"
    rag_answer_events_lanes: tuple = ("multihop", "state")   # lanes that get the schema
                                       # under "lanes" (aggregation + temporal/state)
    # In-text relative-date resolution (query-side, render-time only): annotate relative
    # phrases in episode text with the absolute date they resolve to against the EPISODE's
    # own date ("last week [≈ 2023-03-20] I attended ..."), because the event's date is
    # not the episode's date and the header delta can't express that. Off (default) =
    # context byte-identical to today.
    rag_resolve_reldates: bool = False

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
