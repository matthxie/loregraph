from __future__ import annotations

import os
import sqlite3
import tempfile
from dataclasses import replace

from kg.config import Config
from kg.corpus import CorpusItem
from kg.ingest_cache import (cache_path, ingest_cache_key, save as save_ingest_cache,
                             try_restore as try_restore_ingest_cache)

SESSIONS = [
    CorpusItem(id="s1", modality="text", source_ref="t/1", text="Alice met Bob in Paris.",
              created_at="2024-01-01"),
    CorpusItem(id="s2", modality="text", source_ref="t/2", text="Bob moved to London.",
              created_at="2024-02-01"),
]


def test_key_stable_for_identical_inputs():
    cfg = Config.default()
    k1 = ingest_cache_key("q1", SESSIONS, cfg)
    k2 = ingest_cache_key("q1", list(SESSIONS), cfg)
    assert k1 == k2


def test_key_changes_with_instance_id_or_session_content():
    cfg = Config.default()
    base = ingest_cache_key("q1", SESSIONS, cfg)
    assert ingest_cache_key("q2", SESSIONS, cfg) != base
    changed = [replace_item_text(SESSIONS[0], "Alice met Bob in Rome."), SESSIONS[1]]
    assert ingest_cache_key("q1", changed, cfg) != base


def test_queryside_config_change_does_not_change_key():
    cfg = Config.default()
    base = ingest_cache_key("q1", SESSIONS, cfg)
    queryside_variants = [
        replace(cfg, top_k=999),
        replace(cfg, rerank=not cfg.rerank),
        replace(cfg, rerank_model="some/other-model"),
        replace(cfg, rag_model="gpt-4o"),
        replace(cfg, rag_context_episodes=1),
        replace(cfg, judge_model="gpt-4o"),
        replace(cfg, ppr_damping=0.1),
        replace(cfg, seed_k=1),
        replace(cfg, mmr_lambda=0.1),
        replace(cfg, route=not cfg.route),
        replace(cfg, self_guard="exclude"),
        replace(cfg, self_guard_cap=0.9),
        replace(cfg, community_seed=7),
        replace(cfg, semaphore_limit=1),
        replace(cfg, ingest_flush_every=1),
        replace(cfg, completeness_tier2=not cfg.completeness_tier2),
        replace(cfg, completeness_tier2_model="gpt-4o"),
        replace(cfg, rerank_keep_ppr_top=0),
        replace(cfg, rag_chunks_per_source=1),
    ]
    for variant in queryside_variants:
        assert ingest_cache_key("q1", SESSIONS, variant) == base


def test_ingest_relevant_config_change_changes_key():
    cfg = Config.default()
    base = ingest_cache_key("q1", SESSIONS, cfg)
    ingest_variants = [
        replace(cfg, llm_model="gpt-4o"),
        replace(cfg, extractor_backend="llm"),
        replace(cfg, local_backend="keyword_only"),
        replace(cfg, cue_escalate=not cfg.cue_escalate),
        replace(cfg, embed_model="other/embed"),
        replace(cfg, embed_dim=768),
        replace(cfg, reflexion=not cfg.reflexion),
        replace(cfg, long_doc_chars=1),
        replace(cfg, extract_max_chars=1),
        replace(cfg, extract_max_tokens=1),
        replace(cfg, syn_link_threshold=0.1),
        replace(cfg, syn_merge_threshold=0.1),
        replace(cfg, entropy_min_chars=1),
        replace(cfg, entropy_min_bits=0.1),
        replace(cfg, rel_syn_merge_threshold=0.1),
        replace(cfg, max_relation_labels=1),
        replace(cfg, l3_enabled=not cfg.l3_enabled),
        replace(cfg, self_entity=not cfg.self_entity),
        replace(cfg, shared_edges=not cfg.shared_edges),
        replace(cfg, episode_knn_k=1),
        replace(cfg, chunking="none"),
        replace(cfg, chunk_target_chars=1),
        replace(cfg, chunk_max_chars=1),
        replace(cfg, part_of_weight=0.9),
        replace(cfg, next_weight=0.9),
    ]
    for variant in ingest_variants:
        assert ingest_cache_key("q1", SESSIONS, variant) != base, variant


def test_extractor_prompt_change_invalidates_key(monkeypatch):
    cfg = Config.default()
    base = ingest_cache_key("q1", SESSIONS, cfg)
    from kg.extractors import OpenAIExtractor
    monkeypatch.setattr(OpenAIExtractor, "_SYS", OpenAIExtractor._SYS + " extra")
    assert ingest_cache_key("q1", SESSIONS, cfg) != base


def test_cue_pattern_change_invalidates_key(monkeypatch):
    """Editing kg/cues.py's regex patterns changes which text escalates to the paid
    extractor, which changes what a fresh ingest would write — so it must bust the cache
    even though no Config field or extractor prompt changed. `_extractor_prompt_digest`
    hashes the module's source text (`inspect.getsource`), so we simulate an edit by
    stubbing `getsource` for the `cues` module rather than mutating file content on disk."""
    cfg = Config.default()
    base = ingest_cache_key("q1", SESSIONS, cfg)

    import inspect

    from kg import cues
    real_getsource = inspect.getsource

    def fake_getsource(module):
        if module is cues:
            return real_getsource(module) + "\n# edited\n"
        return real_getsource(module)

    monkeypatch.setattr(inspect, "getsource", fake_getsource)
    assert ingest_cache_key("q1", SESSIONS, cfg) != base


def test_save_and_restore_round_trip():
    with tempfile.TemporaryDirectory() as d:
        store_path = os.path.join(d, "lme_instance.db")
        con = sqlite3.connect(store_path)
        con.execute("CREATE TABLE t(x INTEGER)")
        con.execute("INSERT INTO t VALUES (42)")
        con.commit()
        con.close()

        key = "abc123def456"
        save_ingest_cache(store_path, "q1", key)
        assert os.path.exists(cache_path(store_path, "q1", key))

        os.remove(store_path)
        assert not os.path.exists(store_path)

        hit = try_restore_ingest_cache(store_path, "q1", key)
        assert hit
        con = sqlite3.connect(store_path)
        rows = con.execute("SELECT x FROM t").fetchall()
        con.close()
        assert rows == [(42,)]


def test_restore_miss_when_not_cached():
    with tempfile.TemporaryDirectory() as d:
        store_path = os.path.join(d, "lme_instance.db")
        assert not try_restore_ingest_cache(store_path, "nope", "deadbeefdead")


def replace_item_text(item: CorpusItem, text: str) -> CorpusItem:
    return CorpusItem(id=item.id, modality=item.modality, source_ref=item.source_ref,
                      title=item.title, text=text, image_path=item.image_path,
                      label_hint=item.label_hint, created_at=item.created_at)
