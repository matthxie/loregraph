"""Tests for the test-run dashboard: token/cost metering, the live-token capture path
(via a scripted fake Anthropic client carrying a `.usage`), article-collapsed scoring,
and an offline end-to-end run_testrun that writes the artifact + static dashboard.

Fully offline/deterministic like the rest of the suite. Run: python -m pytest -q
"""
from __future__ import annotations

import json
import os
import tempfile
import types

from kg import Config, KnowledgeGraph
from kg.corpus import CorpusItem
from kg.evaluate import _mrr, _recall_at_k
from kg.metering import UsageMeter, empty_totals, price, totals_of
from kg.testrun import _article, _dedup, run_testrun
from kg import dashboard, testrun


def _cfg() -> Config:
    c = Config.default()
    c.embedder = "hashing"
    c.extractor = "heuristic"
    return c


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


def test_meter_reads_usage_and_offline_is_zero():
    m = UsageMeter()
    # a turn with no .usage (offline / fake stub) records nothing
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
def _graph():
    g = KnowledgeGraph.open(os.path.join(tempfile.mkdtemp(), "kg.db"), _cfg())
    g.ingest([
        CorpusItem(id="a", modality="text", source_ref="u/a", title="Alan Turing",
                   text="Alan Turing worked at Bletchley Park on cryptography and the Enigma."),
        CorpusItem(id="b", modality="text", source_ref="u/b", title="Bletchley Park",
                   text="Bletchley Park was the British codebreaking site; Turing worked there."),
    ])
    return g


def test_claude_agent_populates_usage():
    g = _graph()
    turn = _turn(_tool_use("t1", "submit_answer",
                           {"answer": "Turing worked at Bletchley Park.",
                            "citations": ["obj_a"]}),
                 usage=_usage(1500, 300))
    ans = g.ask("how is Turing connected to Bletchley Park", client=_FakeClient([turn]))
    assert ans.backend == "claude"
    assert ans.usage["llm_calls"] == 1
    assert ans.usage["input_tokens"] == 1500 and ans.usage["output_tokens"] == 300
    assert abs(ans.usage["cost_usd"] - (1500 * 1e-6 + 300 * 5e-6)) < 1e-9


def test_offline_agent_reports_zero_usage():
    g = _graph()
    ans = g.ask("Turing Bletchley", backend="offline")
    assert ans.backend == "offline"
    assert ans.usage["cost_usd"] == 0.0 and ans.usage["llm_calls"] == 0


def test_heuristic_extractor_meter_is_empty():
    g = _graph()                       # offline ingest above
    assert g.extractor.meter.totals()["cost_usd"] == 0.0


# --------------------------------------------------------------------------- #
# article-collapsed scoring (mixed chunk graph vs article-level gold)
# --------------------------------------------------------------------------- #
def test_article_collapse_and_recall():
    assert _article("obj_wiki_062#p003") == "obj_wiki_062"
    assert _article("obj_img_013#p000") == "obj_img_013"
    assert _article("obj_wiki_010") == "obj_wiki_010"   # already article-level: no-op
    ranked = ["obj_wiki_062#p003", "obj_wiki_062#p007", "obj_wiki_005#p001"]
    ranked_art = _dedup(_article(o) for o in ranked)
    assert ranked_art == ["obj_wiki_062", "obj_wiki_005"]   # deduped, order kept
    gold = {"obj_wiki_062"}
    assert _recall_at_k(ranked_art, gold, 8) == 1.0
    assert _mrr(ranked_art, gold) == 1.0
    assert _recall_at_k(_dedup(_article(o) for o in ["obj_wiki_099#p0"]), gold, 8) == 0.0


# --------------------------------------------------------------------------- #
# end-to-end offline run_testrun → artifact + static dashboard
# --------------------------------------------------------------------------- #
def test_run_testrun_offline_writes_artifact():
    tmp = tempfile.mkdtemp()
    cfg = _cfg()
    run = run_testrun(store_path=os.path.join(tmp, "t.db"), limit=8, n_queries=3,
                      backend="offline", judge=False, communities=False,
                      label="t", out_dir=os.path.join(tmp, "runs"), config=cfg)
    # shape
    assert run["backends"]["agent"] == "offline"
    assert run["cost_usd"] == 0.0 and run["tokens"] == 0          # offline → free
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


def test_summarize_runs():
    tmp = tempfile.mkdtemp()
    run = run_testrun(store_path=os.path.join(tmp, "t.db"), limit=6, n_queries=2,
                      backend="offline", judge=False, communities=False,
                      label="s", out_dir=os.path.join(tmp, "runs"), config=_cfg())
    s = testrun.summarize(run)
    assert "INPUT" in s and "QUERY" in s and "TOTAL" in s
