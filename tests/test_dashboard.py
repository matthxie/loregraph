"""Tests for the test-run dashboard: token/cost metering, the live-token capture path
(via a fake Anthropic client carrying a `.usage`), article-collapsed scoring, and an
end-to-end run_testrun that writes the artifact + static dashboard.

The kg library is now LIVE-ONLY (the offline heuristic extractor / hashing embedder /
offline answerer were removed). These tests stay deterministic + free WITHOUT calling
the real Anthropic API by:
  * embedding with the real local bge model (`embedder="st"` — deterministic, no key),
  * stubbing extraction with a `ScriptedExtractor` (a {text: Extraction} table), and
  * injecting a FAKE Anthropic client into the answerer/judge so no network call happens.

Run: python -m pytest tests/test_dashboard.py -q
"""
from __future__ import annotations

import json
import os
import tempfile
import types

import pytest

from kg import Config, KnowledgeGraph
from kg import dashboard, testrun
from kg.corpus import CorpusItem
from kg.embedders import SentenceTransformerEmbedder, get_embedder
from kg.evaluate import _mrr, _recall_at_k
from kg.extractors import (Extraction, ExtractedEntity, ExtractedRelation,
                           ScriptedExtractor, get_extractor)
from kg.metering import UsageMeter, empty_totals, price, totals_of
from kg.models import EntityType, Provenance
from kg.rag import RagAnswerer
from kg.testrun import _article, _dedup, run_per_instance, run_testrun


def _cfg() -> Config:
    c = Config.default()
    c.embedder = "st"          # real local bge — deterministic, free, no key, no network
    return c


# --------------------------------------------------------------------------- #
# Fake Anthropic clients (no network). Two flavours:
#   _FakeAnthropic — the canonical single-turn submit_answer/grade stub used as an
#                    `agent_client` / `judge_client` injection.
#   _FakeClient    — a scripted multi-turn client for the lower-level g.ask() test that
#                    asserts exact usage attribution.
# --------------------------------------------------------------------------- #
class _FakeAnthropic:
    """A fake Anthropic client whose .messages.create returns one tool_use block.
    Works for BOTH the answerer (tool 'submit_answer') and the judge (tool 'grade'):
    it echoes back whatever `input` the caller's tool expects, keyed off tool_choice."""

    def __init__(self, answer="", citations=None, *, correct=True, score=1.0):
        self._answer = answer
        self._citations = citations or []
        self._correct = correct
        self._score = score
        self.messages = self
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        name = (kw.get("tool_choice") or {}).get("name", "submit_answer")
        if name == "grade":
            inp = {"correct": self._correct, "score": self._score, "reason": "ok"}
        else:
            inp = {"answer": self._answer, "citations": list(self._citations)}
        blk = types.SimpleNamespace(type="tool_use", name=name, input=inp)
        usage = types.SimpleNamespace(input_tokens=0, output_tokens=0,
                                      cache_read_input_tokens=0,
                                      cache_creation_input_tokens=0)
        return types.SimpleNamespace(content=[blk], usage=usage, stop_reason="tool_use")


def _usage(i, o, cr=0, cw=0):
    return types.SimpleNamespace(input_tokens=i, output_tokens=o,
                                 cache_read_input_tokens=cr, cache_creation_input_tokens=cw)


def _turn(*blocks, usage=None):
    return types.SimpleNamespace(content=list(blocks), stop_reason="tool_use", usage=usage)


def _tool_use(tid, name, inp):
    return types.SimpleNamespace(type="tool_use", id=tid, name=name, input=inp)


class _FakeClient:
    def __init__(self, script):
        self._script = list(script)
        self.messages = self
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self._script.pop(0)


# Scripted extractions for the two demo episodes, rich enough that mentions /
# SHARED_ENTITY and entity-anchored retrieval still behave like the live graph.
_TURING_TEXT = "Alan Turing worked at Bletchley Park on cryptography and the Enigma."
_BLETCHLEY_TEXT = "Bletchley Park was the British codebreaking site; Turing worked there."

_SCRIPT = {
    _TURING_TEXT: Extraction(
        entities=[ExtractedEntity("Alan Turing", EntityType.PERSON),
                  ExtractedEntity("Bletchley Park", EntityType.PLACE),
                  ExtractedEntity("Enigma", EntityType.CONCEPT)],
        tags=["cryptography", "codebreaking", "enigma", "world war ii"],
        relations=[ExtractedRelation(source="Alan Turing", target="Bletchley Park",
                                     labels=["worked_at"], provenance=Provenance.EXTRACTED,
                                     confidence=0.9)],
    ),
    _BLETCHLEY_TEXT: Extraction(
        entities=[ExtractedEntity("Bletchley Park", EntityType.PLACE),
                  ExtractedEntity("Alan Turing", EntityType.PERSON)],
        tags=["codebreaking", "cryptography", "britain"],
        relations=[ExtractedRelation(source="Alan Turing", target="Bletchley Park",
                                     labels=["worked_at"], provenance=Provenance.EXTRACTED,
                                     confidence=0.9)],
    ),
}


def _scripted_graph() -> KnowledgeGraph:
    """A two-episode graph ingested through a ScriptedExtractor — no live extraction."""
    g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), _cfg())
    g.extractor = ScriptedExtractor(_SCRIPT)       # stub the LLM extraction step
    g.ingest([
        CorpusItem(id="a", modality="text", source_ref="u/a", title="Alan Turing",
                   text=_TURING_TEXT),
        CorpusItem(id="b", modality="text", source_ref="u/b", title="Bletchley Park",
                   text=_BLETCHLEY_TEXT),
    ])
    return g


# --------------------------------------------------------------------------- #
# factory behaviour (live-only): the removed offline backends are now replaced by
# the new contract — get_embedder always returns bge, get_extractor RAISES w/o key,
# RagAnswerer RAISES with neither a client nor a key.
# --------------------------------------------------------------------------- #
def test_get_embedder_is_sentence_transformer():
    emb = get_embedder(_cfg())
    assert isinstance(emb, SentenceTransformerEmbedder)
    assert emb.name.startswith("st:")


def test_haiku_backend_requires_key(monkeypatch):
    # default 'cue_gated' is keyless; the live 'haiku' backend still RAISES without a key
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = _cfg(); c.extractor_backend = "haiku"
    with pytest.raises(RuntimeError):
        get_extractor(c)


def test_answerer_requires_client_or_key(monkeypatch):
    """No injected client + no key => the live-only answerer refuses to silently
    degrade (the offline answerer was removed)."""
    g = _scripted_graph()                              # build while the key is present
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        RagAnswerer(g.store, g.embedder, g.canon, _cfg(), client=None)


# --------------------------------------------------------------------------- #
# metering math
# --------------------------------------------------------------------------- #
def test_pricing_table():
    # haiku: 1/5 per MTok
    assert abs(price("claude-haiku-4-5-20251001", 1_000_000, 0) - 1.0) < 1e-9
    assert abs(price("claude-haiku-4-5-20251001", 0, 1_000_000) - 5.0) < 1e-9
    # opus: 5/25
    assert abs(price("claude-opus-4-8", 1_000_000, 1_000_000) - 30.0) < 1e-9
    # unknown model falls back to haiku rate (never silently free)
    assert price("made-up-model", 1_000_000, 0) > 0


def test_meter_reads_usage_and_no_usage_is_zero():
    m = UsageMeter()
    # a turn with no .usage (a bare stub) records nothing
    assert m.record("agent", "claude-haiku-4-5-20251001", types.SimpleNamespace()) is None
    assert m.totals()["cost_usd"] == 0.0
    # a real-looking usage is priced
    m.record("extract", "claude-haiku-4-5-20251001", _turn(usage=_usage(1000, 200)))
    t = m.totals()
    assert t["llm_calls"] == 1 and t["input_tokens"] == 1000 and t["output_tokens"] == 200
    assert abs(t["cost_usd"] - (1000 * 1e-6 + 200 * 5e-6)) < 1e-9
    assert empty_totals()["cost_usd"] == 0.0 and totals_of([])["tokens"] == 0


# --------------------------------------------------------------------------- #
# live-token capture path (the whole reason the dashboard can show real cost)
# --------------------------------------------------------------------------- #
def test_claude_agent_populates_usage():
    g = _scripted_graph()
    turn = _turn(_tool_use("t1", "submit_answer",
                           {"answer": "Turing worked at Bletchley Park.",
                            "citations": ["ep_a"]}),
                 usage=_usage(1500, 300))
    ans = g.ask("how is Turing connected to Bletchley Park", client=_FakeClient([turn]))
    assert ans.backend == "claude"
    assert ans.usage["llm_calls"] == 1
    assert ans.usage["input_tokens"] == 1500 and ans.usage["output_tokens"] == 300
    assert abs(ans.usage["cost_usd"] - (1500 * 1e-6 + 300 * 5e-6)) < 1e-9


def test_fake_client_with_zero_usage_reports_zero_cost():
    """The canonical _FakeAnthropic stub carries a zero-token .usage, so the meter
    records a (free) call: backend is the live 'claude' answerer, cost is $0."""
    g = _scripted_graph()
    ans = g.ask("Turing Bletchley", client=_FakeAnthropic(answer="Turing was at Bletchley.",
                                                          citations=["ep_a"]))
    assert ans.backend == "claude"
    assert ans.usage["cost_usd"] == 0.0


def test_scripted_extractor_meter_is_empty():
    g = _scripted_graph()              # scripted (stubbed) ingest above — no live calls
    assert g.extractor.meter.totals()["cost_usd"] == 0.0


# --------------------------------------------------------------------------- #
# article-collapsed scoring (mixed chunk graph vs article-level gold)
# --------------------------------------------------------------------------- #
def test_article_collapse_and_recall():
    # _article strips both the chunk suffix AND the node prefix, so episode ids (ep_) match
    # gold authored under the old object prefix (obj_).
    assert _article("ep_wiki_062#p003") == "wiki_062"
    assert _article("obj_wiki_062#p003") == "wiki_062"   # cross-rename: same article
    assert _article("ep_img_013#p000") == "img_013"
    assert _article("obj_wiki_010") == "wiki_010"        # already article-level: just de-prefix
    ranked = ["ep_wiki_062#p003", "ep_wiki_062#p007", "ep_wiki_005#p001"]
    ranked_art = _dedup(_article(o) for o in ranked)
    assert ranked_art == ["wiki_062", "wiki_005"]        # deduped, order kept
    gold = {_article(x) for x in {"obj_wiki_062"}}        # gold authored under the old prefix
    assert _recall_at_k(ranked_art, gold, 8) == 1.0
    assert _mrr(ranked_art, gold) == 1.0
    assert _recall_at_k(_dedup(_article(o) for o in ["ep_wiki_099#p0"]), gold, 8) == 0.0


# --------------------------------------------------------------------------- #
# end-to-end run_testrun → artifact + static dashboard (no live API)
# --------------------------------------------------------------------------- #
def _patch_offline_extraction(monkeypatch):
    """Make the graph that run_testrun builds use an (empty-table) ScriptedExtractor, so
    its internal ingest writes + bge-embeds episodes WITHOUT a live Haiku call. Episodes
    still land (with their ids), so all the structural assertions still hold."""
    monkeypatch.setattr("kg.graph.get_extractor", lambda cfg: ScriptedExtractor({}))


def test_run_testrun_writes_artifact(monkeypatch):
    _patch_offline_extraction(monkeypatch)
    tmp = tempfile.mkdtemp()
    cfg = _cfg()
    run = run_testrun(store_path=os.path.join(tmp, "t.db"), tier="sample",
                      limit=8, n_queries=3,
                      backend=None, judge=False, communities=False,
                      label="t", out_dir=os.path.join(tmp, "runs"), config=cfg,
                      agent_client=_FakeAnthropic(answer="An answer.", citations=[]),
                      judge_client=_FakeAnthropic())
    # shape — the agent backend is the live 'claude' answerer (over the fake client)
    assert run["backends"]["agent"] == "claude"
    assert len(run["ingest"]["steps"]) == 8
    assert run["ingest"]["graph"]["nodes"] and "build_order" in run["ingest"]["graph"]
    assert run["ingest"]["totals"]["nodes"] > 0
    q = run["query"]["queries"]
    assert len(q) == 3
    for r in q:
        assert "subgraph" in r and "trace" in r and "gold_marks" in r
        assert "recall_at_k" in r and "cost_usd" in r
    # files written
    run_dir = os.path.join(tmp, "runs", "t")
    assert os.path.exists(os.path.join(run_dir, "run.json"))
    html = open(os.path.join(run_dir, "dashboard.html"), encoding="utf-8").read()
    assert len(html) > 5000 and "Input" in html and "Query" in html
    idx = json.load(open(os.path.join(tmp, "runs", "index.json"), encoding="utf-8"))
    assert idx and idx[0]["run_id"] == "t"
    # round-trips through the index/run renderers without error
    assert "<svg" in dashboard.render_run_html(run) or "svg" in dashboard.render_run_html(run)
    assert dashboard.render_index_html(idx)


def test_summarize_runs(monkeypatch):
    _patch_offline_extraction(monkeypatch)
    tmp = tempfile.mkdtemp()
    run = run_testrun(store_path=os.path.join(tmp, "t.db"), tier="sample",
                      limit=6, n_queries=2,
                      backend=None, judge=False, communities=False,
                      label="s", out_dir=os.path.join(tmp, "runs"), config=_cfg(),
                      agent_client=_FakeAnthropic(answer="An answer.", citations=[]),
                      judge_client=_FakeAnthropic())
    s = testrun.summarize(run)
    assert "INPUT" in s and "QUERY" in s and "TOTAL" in s


# --------------------------------------------------------------------------- #
# per-instance LongMemEval protocol (fresh graph per question, no cross-user pooling)
# --------------------------------------------------------------------------- #
def test_run_per_instance_isolated_and_reconciled(monkeypatch):
    import re
    _patch_offline_extraction(monkeypatch)
    tmp = tempfile.mkdtemp()
    run = run_per_instance(tier="sample", store_path=os.path.join(tmp, "t.db"),
                           n_queries=4, backend=None, judge=False, communities=False,
                           label="pi", out_dir=os.path.join(tmp, "runs"), config=_cfg(),
                           agent_client=_FakeAnthropic(answer="An answer.", citations=[]),
                           judge_client=_FakeAnthropic())
    assert run["config"]["mode"] == "per-instance"
    q = run["query"]["queries"]
    assert len(q) == 4
    # cost reconciles: top-level == ingest + query (identity holds regardless of magnitude)
    it, qt = run["ingest"]["totals"], run["query"]["totals"]
    assert run["cost_usd"] == round(it["cost_usd"] + qt["cost_usd"], 6)
    # cumulative graph size == sum of per-instance contributions
    assert it["nodes"] == sum(s["added_nodes"] for s in run["ingest"]["steps"])
    # ISOLATION: every retrieved/seed/cited episode id namespaces to its OWN question_id
    for r in q:
        for oid in r["object_ids"] + r.get("seeds", []) + r.get("citations", []):
            if oid.startswith("ep_"):
                m = re.match(r"ep_(.+?)__", oid)
                assert m and m.group(1) == r["id"], f"cross-instance leak: {oid} in {r['id']}"
    # the Input view's representative graph is a real (labeled) single instance
    g = run["ingest"]["graph"]
    assert g.get("representative_of") and g["nodes"]
    # the static dashboard renders for a per-instance run
    html = dashboard.render_run_html(run)
    assert len(html) > 5000 and "per-instance" in html


def test_per_instance_queries_zero_means_none():
    from kg.corpus import iter_lme_instances
    assert list(iter_lme_instances("sample", limit=0)) == []
