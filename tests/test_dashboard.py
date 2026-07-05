"""Tests for the test-run dashboard: token/cost metering, the live-token capture path
(via a fake OpenAI client carrying a `.usage`), article-collapsed scoring, and an
end-to-end run_testrun that writes the artifact + static dashboard.

The kg library is now LIVE-ONLY (the offline heuristic extractor / hashing embedder /
offline answerer were removed). These tests stay deterministic + free WITHOUT calling
the real OpenAI API by:
  * embedding with the real local bge model (`embedder="st"` — deterministic, no key),
  * stubbing extraction with a `ScriptedExtractor` (a {text: Extraction} table), and
  * injecting a FAKE OpenAI client into the answerer/judge so no network call happens.

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
# Fake OpenAI clients (no network). Two flavours:
#   _FakeAnthropic — the canonical single-turn submit_answer/grade stub used as an
#                    `agent_client` / `judge_client` injection. (Name kept for the
#                    historical "fake external LLM client" role; it now speaks the
#                    OpenAI chat.completions shape, matching kg.rag.OpenAIAnswerer
#                    and kg.testrun._judge.)
#   _FakeClient    — a scripted multi-turn client for the lower-level g.ask() test that
#                    asserts exact usage attribution.
# --------------------------------------------------------------------------- #
class _FakeAnthropic:
    """A fake OpenAI client whose .chat.completions.create returns one tool call.
    Works for BOTH the answerer (tool 'submit_answer') and the judge (tool 'grade'):
    it echoes back whatever arguments the caller's tool expects, keyed off tool_choice."""

    def __init__(self, answer="", citations=None, *, correct=True, score=1.0):
        self._answer = answer
        self._citations = citations or []
        self._correct = correct
        self._score = score
        self.chat = self
        self.completions = self
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        name = ((kw.get("tool_choice") or {}).get("function") or {}).get("name", "submit_answer")
        if name == "grade":
            args = {"correct": self._correct, "score": self._score, "reason": "ok"}
        else:
            args = {"answer": self._answer, "citations": list(self._citations)}
        tc = types.SimpleNamespace(id="call_0", function=types.SimpleNamespace(
            name=name, arguments=json.dumps(args)))
        message = types.SimpleNamespace(content=None, tool_calls=[tc])
        choice = types.SimpleNamespace(message=message, finish_reason="tool_calls")
        usage = types.SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        return types.SimpleNamespace(choices=[choice], usage=usage)


def _usage(i, o):
    return types.SimpleNamespace(prompt_tokens=i, completion_tokens=o)


def _turn(*tool_calls, usage=None):
    message = types.SimpleNamespace(content=None, tool_calls=list(tool_calls) or None)
    choice = types.SimpleNamespace(message=message,
                                   finish_reason="tool_calls" if tool_calls else "stop")
    return types.SimpleNamespace(choices=[choice], usage=usage)


def _tool_use(tid, name, inp):
    return types.SimpleNamespace(id=tid, function=types.SimpleNamespace(
        name=name, arguments=json.dumps(inp)))


class _FakeClient:
    def __init__(self, script):
        self._script = list(script)
        self.chat = self
        self.completions = self
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
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = _cfg(); c.extractor_backend = "haiku"
    with pytest.raises(RuntimeError):
        get_extractor(c)


def test_answerer_requires_client_or_key(monkeypatch):
    """No injected client + no key => the live-only answerer refuses to silently
    degrade (the offline answerer was removed)."""
    g = _scripted_graph()                              # build while the key is present
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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
def test_openai_agent_populates_usage():
    g = _scripted_graph()
    turn = _turn(_tool_use("t1", "submit_answer",
                           {"answer": "Turing worked at Bletchley Park.",
                            "citations": ["ep_a"]}),
                 usage=_usage(1500, 300))
    ans = g.ask("how is Turing connected to Bletchley Park", client=_FakeClient([turn]))
    assert ans.backend == "openai"
    assert ans.usage["llm_calls"] == 1
    assert ans.usage["input_tokens"] == 1500 and ans.usage["output_tokens"] == 300
    # rag_model defaults to gpt-4o-mini: $0.15/$0.60 per MTok in/out
    assert abs(ans.usage["cost_usd"] - (1500 * 0.15e-6 + 300 * 0.60e-6)) < 1e-9


def test_fake_client_with_zero_usage_reports_zero_cost():
    """The canonical _FakeAnthropic stub carries a zero-token .usage, so the meter
    records a (free) call: backend is the live 'openai' answerer, cost is $0."""
    g = _scripted_graph()
    ans = g.ask("Turing Bletchley", client=_FakeAnthropic(answer="Turing was at Bletchley.",
                                                          citations=["ep_a"]))
    assert ans.backend == "openai"
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
    # shape — the agent backend is the live 'openai' answerer (over the fake client)
    assert run["backends"]["agent"] == "openai"
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
    # profiler: stage timing + cost-by-site land in the run and per-item records
    prof = run["profile"]
    assert set(prof) == {"ingest", "query", "cost_by_site"}
    assert any(k.startswith("ingest.") for k in prof["ingest"])       # pipeline wall stages
    assert all(v["seconds"] >= 0 and v["calls"] >= 1 for v in prof["ingest"].values())
    assert any(k.startswith("query.") for k in prof["query"])         # ask() stages
    assert "rag" in prof["cost_by_site"]                              # answer-call site
    for s in run["ingest"]["steps"]:
        assert "profile" in s                                         # per-step breakdown
    for r in q:
        assert "profile" in r and r["seconds"] >= 0                   # per-query breakdown
    assert idx[0]["ingest_seconds"] is not None                       # run-to-run comparison


def test_profiler_spans_and_ambient_activation():
    """kg/profiler.py: spans aggregate per label into the ACTIVE profiler; with none
    active span() is a no-op; drain() resets; merge_profiles accumulates label-wise."""
    from kg.profiler import (Profiler, activate, compact, deactivate,
                             merge_profiles, span)

    with span("orphan.stage"):        # no active profiler -> silently does nothing
        pass
    p = Profiler()
    activate(p)
    try:
        with span("stage.a"):
            pass
        with span("stage.a"):
            pass
        with span("stage.b"):
            pass
    finally:
        deactivate()
    snap = p.snapshot()
    assert snap["stage.a"]["calls"] == 2 and snap["stage.b"]["calls"] == 1
    assert all(v["seconds"] >= 0 for v in snap.values())
    with span("stage.late"):          # deactivated -> not recorded
        pass
    assert "stage.late" not in p.snapshot()
    total: dict = {}
    merge_profiles(total, p.drain())
    merge_profiles(total, {"stage.a": {"seconds": 1.0, "calls": 3}})
    assert total["stage.a"]["calls"] == 5
    assert p.snapshot() == {}         # drained
    assert compact({"x": {"seconds": 0.5, "calls": 2}}) == {"x": 0.5}


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


# --------------------------------------------------------------------------- #
# extraction-completeness wiring (kg/completeness.py; see spikes/completeness/REPORT.md)
# --------------------------------------------------------------------------- #
class _FakeOccurrenceClient:
    """Fake chat.completions client for the tier-2 occurrence-enumeration call: always
    reports zero occurrences, so tier 2 runs (cost/meter path exercised) without needing
    the ScriptedExtractor's synthetic graph to resemble a real audit target."""

    def __init__(self):
        self.chat = self
        self.completions = self
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        message = types.SimpleNamespace(
            content=json.dumps({"occurrences": [], "notes": "none found"}))
        choice = types.SimpleNamespace(message=message)
        usage = types.SimpleNamespace(prompt_tokens=50, completion_tokens=10)
        return types.SimpleNamespace(choices=[choice], usage=usage)


def test_run_per_instance_completeness_tier1_only(monkeypatch):
    """tier 1 (regex capture rate) runs for free on the sample tier's aggregate-shaped
    questions even with tier 2 off; questions without a parseable quantity are skipped
    (not counted as a misleading 0)."""
    _patch_offline_extraction(monkeypatch)
    tmp = tempfile.mkdtemp()
    cfg = _cfg()
    cfg.completeness_tier2 = False
    run = run_per_instance(tier="sample", store_path=os.path.join(tmp, "t.db"),
                           n_queries=None, backend=None, judge=False, communities=False,
                           label="c1", out_dir=os.path.join(tmp, "runs"), config=cfg,
                           agent_client=_FakeAnthropic(answer="An answer.", citations=[]))
    cmpl = run["completeness"]
    assert cmpl["tier2"]["enabled"] is False and cmpl["tier2"]["n"] is None
    # the sample tier's "$ coffee mug" question (0100672e) has evidence-text amounts, so
    # tier 1 should have produced at least one per-question record.
    t1 = cmpl["tier1"]
    assert t1 is not None
    assert t1["n_questions"] >= 1
    assert 0.0 <= t1["capture_rate"] <= 1.0
    ids = {r["question_id"] for r in t1["per_question"]}
    assert "0100672e" in ids
    # index.json carries the headline number for the cross-run trend
    idx = json.load(open(os.path.join(tmp, "runs", "index.json"), encoding="utf-8"))
    assert idx[0]["completeness_tier1_capture_rate"] == t1["capture_rate"]
    # dashboard renders without error either way
    html = dashboard.render_run_html(run)
    assert "extraction completeness" in html


def test_run_per_instance_completeness_tier2_runs_and_meters(monkeypatch):
    """tier 2 fires its LLM call for aggregate questions with gold evidence and records
    cost under the 'audit.completeness' site, even when it finds zero occurrences."""
    _patch_offline_extraction(monkeypatch)
    tmp = tempfile.mkdtemp()
    cfg = _cfg()
    cfg.completeness_tier2 = True
    fake_audit = _FakeOccurrenceClient()
    run = run_per_instance(tier="sample", store_path=os.path.join(tmp, "t.db"),
                           n_queries=None, backend=None, judge=False, communities=False,
                           label="c2", out_dir=os.path.join(tmp, "runs"), config=cfg,
                           agent_client=_FakeAnthropic(answer="An answer.", citations=[]),
                           completeness_client=fake_audit)
    assert len(fake_audit.calls) >= 1                       # one per aggregate question
    assert run["profile"]["cost_by_site"]["audit.completeness"]["llm_calls"] >= 1
    assert run["completeness"]["tier2"]["enabled"] is True


def test_completeness_none_when_no_aggregate_questions(monkeypatch):
    """A tier with no aggregate-shaped questions (or none with gold evidence) must show
    n/a (None), never a misleading 0 — exercised here via a --limit that starves the
    per-instance loop of any instance at all."""
    _patch_offline_extraction(monkeypatch)
    tmp = tempfile.mkdtemp()
    run = run_per_instance(tier="sample", store_path=os.path.join(tmp, "t.db"),
                           n_queries=0, backend=None, judge=False, communities=False,
                           label="c3", out_dir=os.path.join(tmp, "runs"), config=_cfg(),
                           agent_client=_FakeAnthropic(answer="An answer.", citations=[]))
    assert run["completeness"]["tier1"] is None
    assert run["completeness"]["tier2"]["n"] is None
