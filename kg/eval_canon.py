"""Canonicalization gate — the eval that must pass BEFORE enabling the L3 tie-breaker.

Feeds hand-labeled predicate and entity pairs through the canonicalizer and checks
that synonyms merge while antonyms / inverses / distinct senses stay separate. The
single load-bearing metric is `wrong_antonym_inverse_merges == 0`: over-merging an
inverse/antonym is the asymmetrically-harmful failure the whole design guards against.

Runs fully offline on the deterministic L1/L2 path (hashing/bge embedder). Add `--l3`
(needs ANTHROPIC_API_KEY) to also exercise the LLM tie-breaker.

    python -m kg eval-canon                          # deterministic path, hashing/bge
    python -m kg eval-canon --embedder st            # real bge-small embeddings
    python -m kg eval-canon --l3 --model claude-haiku-4-5-20251001   # + L3 adjudicator
"""
from __future__ import annotations

from .canonicalize import Canonicalizer
from .config import Config
from .embedders import get_embedder
from .models import EntityType, Modality, episode_node
from .store import GraphStore

# (a, b, should_merge, category)
PREDICATE_PAIRS = [
    ("is_friend_of", "is_friends_with", True, "synonym-inflection"),
    ("works_with", "works with", True, "synonym-surface"),
    ("located_in", "located-in", True, "synonym-surface"),
    ("works_with", "collaborates_with", True, "synonym-lexical"),
    ("located_in", "situated_in", True, "synonym-lexical"),
    ("founded", "established", True, "synonym-lexical"),
    ("is_friend_of", "is_enemy_of", False, "antonym"),
    ("ally_of", "rival_of", False, "antonym"),
    ("predecessor_of", "successor_of", False, "antonym"),
    ("parent_of", "child_of", False, "inverse"),
    ("manages", "managed_by", False, "passive-inverse"),
    ("founded", "founded_by", False, "passive-inverse"),
    ("employs", "employed_by", False, "passive-inverse"),
    ("founded", "located_in", False, "unrelated"),
    ("manages", "discovered", False, "unrelated"),
]

ENTITY_PAIRS = [
    ("United States of America", "USA", True, "alias"),
    ("Barack Obama", "Obama", True, "alias"),
    ("natural language processing", "NLP", True, "alias"),
    ("Paris", "Paris, Texas", False, "distinct-place"),
    ("Apple Inc.", "apple", False, "distinct-sense"),
    ("Georgia", "George", False, "distinct"),
]

_CRITICAL = {"antonym", "inverse", "passive-inverse"}


def _fresh(cfg: Config) -> Canonicalizer:
    """A canonicalizer over a fresh store seeded with a few object nodes (so IDF has a
    non-trivial corpus size). Each pair is evaluated in isolation so earlier pairs don't
    pollute later ones."""
    store = GraphStore(cfg)
    for i in range(10):
        store.add_node(episode_node(f"ep_{i}", modality=Modality.TEXT, source_ref="u",
                                    raw_text="x", content_hash=str(i), ts="t"))
    return Canonicalizer(store, get_embedder(cfg), cfg)


def _run_pairs(cfg: Config, pairs: list[tuple], kind: str) -> dict:
    tp = fp = fn = tn = 0
    wrong_critical = 0
    rows = []
    for a, b, should, cat in pairs:
        canon = _fresh(cfg)
        if kind == "relation":
            ia = canon.resolve_relation(a)
            ib = canon.resolve_relation(b)
        else:
            ia = canon.resolve_entity(a, EntityType.OTHER)
            ib = canon.resolve_entity(b, EntityType.OTHER)
        merged = (ia == ib)
        if should and merged:
            tp += 1
        elif should and not merged:
            fn += 1
        elif (not should) and merged:
            fp += 1
            if cat in _CRITICAL:
                wrong_critical += 1
        else:
            tn += 1
        rows.append({"a": a, "b": b, "category": cat,
                     "should_merge": should, "merged": merged,
                     "ok": merged == should})
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "kind": kind,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "wrong_antonym_inverse_merges": wrong_critical,
        "false_merges": fp,
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "pairs": rows,
    }


def run_gate(cfg: Config) -> dict:
    rel = _run_pairs(cfg, PREDICATE_PAIRS, "relation")
    ent = _run_pairs(cfg, ENTITY_PAIRS, "entity")
    # The gate passes iff no inverse/antonym predicate pair wrongly merged AND no
    # distinct-sense/place entity pair wrongly merged. (Synonym RECALL is reported but
    # not gated — it depends on the embedder/L3 and is the upside, not the safety risk.)
    gate_pass = (rel["wrong_antonym_inverse_merges"] == 0 and ent["false_merges"] == 0)
    return {
        "config": {
            "extractor": cfg.extractor,
            "embedder": cfg.embedder,
            "embed_model": cfg.embed_model,
            "l3_enabled": cfg.l3_enabled,
            "l3_model": cfg.l3_model if cfg.l3_enabled else None,
        },
        "gate_pass": gate_pass,
        "relation": rel,
        "entity": ent,
    }
