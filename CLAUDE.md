<!-- TEMPORARY scratch reference — extraction/canonicalization prompt pipeline.
     Safe to delete once the flow is internalized; not part of the permanent docs.
     Authoritative design lives in docs/ARCHITECTURE.md §2/§3/§6. -->

# TEMP: How prompts are applied to the data

This is the order in which LLM prompts touch each ingested object, and where the
"check against existing entity/relation tags" actually happens. **The extraction
prompt is blind** — it never sees the existing graph vocabulary. Checking against
existing tags is a *separate, downstream* concern (deterministic first, LLM only on
the ambiguous residual).

## Per-object pipeline (`kg/ingest.py` → `Ingestor.ingest`)

```
INTAKE → CACHE/supersede → NORMALIZE → EXTRACT → (CANONICALIZE = check vs existing) → EMBED → WRITE → DERIVE edges
                                         └── prompts live here ──┘   └── L3 prompt here ──┘
```

### The prompts, in firing order

| # | Prompt | Where | Fires | Sees existing vocab? |
|---|--------|-------|-------|----------------------|
| 1 | **Extraction** (`emit_graph` tool call) | `HaikuExtractor._SYS` + `GRAPH_TOOL` | once per object (long docs: once per ~6k-char section, unioned) | **No** — blind, static, `temperature=0` |
| 2 | **Reflexion** (recall) | `HaikuExtractor._reflexion` | once, right after #1 (`config.reflexion`, default on) | No — same static system prompt; "list only what you OMITTED" |
| — | **Canonicalize: L1 + L2** (no LLM) | `Canonicalizer.resolve_tag/entity/relation` | during the sequential WRITE loop, per surface | **Yes** — this is the primary "check vs existing": L1 normalized/content-key hash → L2 bge-small cosine merge |
| 3 | **L3 tie-breaker** (merge-or-new) | `Canonicalizer._l3_adjudicate` | only on gray-band candidates, **only if `l3_enabled`** (default **OFF**) | **Yes** — shown only the ~5 nearest existing canonical labels (IDF-ranked) |

So per object the LLM runs **1 extraction call + 1 reflexion call**, then deterministic
canonicalization, then **0–N tiny L3 calls** (only for ambiguous gray-band surfaces, and
only when enabled). Extraction and "check vs existing" are never fused into one prompt.

### Where "check against existing tags" lives (two tiers)

- **Tier A — deterministic (`canonicalize.py`, always on, no LLM):** does ~all of the
  checking. `resolve_*` compare each new surface to every existing node via
  - **L1** exact key: `normalize_key` (tags/entities) / `relation_content_key` (predicates —
    drops function words, singularizes, **keeps passive `by`** so `manages` ≠ `managed_by`).
  - **L2** bge-small cosine merge: tags/entities ≥ `0.93`, relations ≥ `0.95` (merge-only,
    **no** SIMILAR_TO links between predicates — antonyms sit close in embedding space).
    Entropy guard blocks short/low-entropy strings ("AI","US") from fuzzy merge.
- **Tier B — L3 LLM tie-breaker (off by default):** fires only on the *gray band* the
  deterministic tiers can't confidently resolve — tags/entities in `[0.85, 0.93)`,
  relations in `[0.90, 0.95)` — and for relations **only after a deterministic
  antonym/inverse/passive veto** (`relation_merge_vetoed`) removes the dangerous pairs so
  the model can never wrongly merge an inverse. Decided *inside* `resolve_*` before any
  node is minted, so a MERGE just returns the existing id (no provisional node, no edge
  rewrite). Under-merge default on any error/parse failure.

### Why not inject the vocabulary into the extraction prompt
Anchoring/over-merge bias, unbounded-vocabulary scale, ingest-order non-determinism, and
it duplicates Tier A. (Prompt-cache is a *future* factor too: the prefix is ~1k tokens,
under Haiku 4.5's 4096-token cache minimum, so caching is moot until the prefix grows.)
Consensus across EDC, AutoSchemaKG, GraphRAG, iText2KG, KGGen, graphiti, mem0-v3.

## Building the per-mode tag dataset (compare models)

`extract-dump` runs **only** the extraction step (no graph, no canonicalization) and
writes one JSONL record per object + a `.summary.json` vocabulary aggregate, so you can
diff what each model produces. `--model` sets the LLM (needs `ANTHROPIC_API_KEY`).

```bash
# offline baseline (no key needed)
python -m kg extract-dump --extractor heuristic              --out store/dump_heuristic.jsonl

# per-model (set ANTHROPIC_API_KEY first)
python -m kg extract-dump --extractor haiku --model claude-haiku-4-5-20251001 --n-text 30 --out store/dump_haiku.jsonl
python -m kg extract-dump --extractor haiku --model claude-sonnet-4-6         --n-text 30 --out store/dump_sonnet.jsonl
python -m kg extract-dump --extractor haiku --model claude-opus-4-8           --n-text 30 --out store/dump_opus.jsonl
# add images with --n-image N   (default 0 = text only; 0 means none of that modality)
```

Each `*.summary.json` has: `unique_tags`, `top_tags`, `entity_types`, `unique_entities`,
`unique_relation_labels`, `top_relation_labels`, `failed`. Diff those across modes to see
how tag/entity/relation vocabularies differ by model.

## The gate before enabling L3

`eval-canon` feeds hand-labeled predicate/entity pairs through the canonicalizer. **The
gate passes iff zero antonym/inverse/distinct-sense pairs wrongly merge.** Synonym
*recall* is reported but not gated (that's L3's upside, not a safety risk).

```bash
python -m kg eval-canon                 # deterministic path (hashing) — fast
python -m kg eval-canon --embedder st   # real bge-small embeddings
python -m kg eval-canon --l3 --model claude-haiku-4-5-20251001   # + L3 (needs key)
```

Verified today: GATE PASS under both hashing and bge — `wrong_antonym_inverse_merges=0`.
Enable L3 in a run with `python -m kg ingest --l3 ...` only after this gate passes and the
extraction (Haiku) path is validated live (it is currently untested — no key on this box).

## Files touched
- `kg/extractors.py` — new blind `_SYS`, `temperature=0`, forced-commitment `_reflexion`, shared `extract_text_sectioned`.
- `kg/canonicalize.py` — `relation_merge_vetoed`, `_l3*` adjudicator, gray-band path in `resolve_tag/entity/relation`.
- `kg/config.py` — `l3_enabled` (off), `l3_model`, `rel_gray_floor`.
- `kg/extract_dump.py`, `kg/eval_canon.py` — new tools. `kg/cli.py` — `extract-dump`, `eval-canon`, `--model`, `--l3`.
