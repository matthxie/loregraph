"""Oracle-leak guard: benchmark metadata must never influence the answer pipeline.

The LongMemEval dataset labels every question with a `kind` (question type), gold
evidence ids, and an expected answer. Any of these flowing into routing, retrieval, or
answering makes benchmark numbers unreproducible in production, where only the question
text and the current time exist. This happened once: route() accepted the dataset `kind`
and mapped it straight to a retrieval lane, silently inflating every lane-gated feature
(69/100 questions routed differently without it). These tests make the whole class of
leak a hard failure instead of a code-review catch.

Allowed inputs to the pipeline: the question text and `as_of` (the question's "now" —
production knows the current time). Everything else in the dataset record is scoring
material only.
"""
from __future__ import annotations

import inspect
import re

from kg.graph import KnowledgeGraph
from kg.rag import RagAnswerer
from kg.retrieval import HybridRetriever
from kg.route import route

FORBIDDEN_PARAMS = {"kind", "question_type", "gold", "gold_ids", "answer_expected",
                    "expected", "label", "lane_hint"}


def _params(fn) -> set:
    return set(inspect.signature(fn).parameters)


def test_router_reads_query_text_only():
    assert _params(route) == {"query"}, (
        "route() must classify from the question text alone — no dataset kind, no hints")


def test_pipeline_signatures_reject_oracle_params():
    for fn in (KnowledgeGraph.ask, RagAnswerer.run, HybridRetriever.retrieve):
        leaked = _params(fn) & FORBIDDEN_PARAMS
        assert not leaked, f"{fn.__qualname__} accepts oracle parameter(s): {leaked}"


def test_testrun_never_passes_dataset_fields_into_ask():
    """The harness may read q['kind']/q['gold'] for SCORING, but the g.ask(...) calls it
    makes must receive nothing from the dataset record except the question text and
    question_date (the as-of anchor)."""
    import kg.testrun as testrun
    src = inspect.getsource(testrun)
    for call in re.findall(r"g\.ask\((?:[^()]|\([^()]*\))*\)", src):
        for field in ("kind", "gold", "answer_expected", "rationale"):
            assert f"q.get(\"{field}\")" not in call and f"q[\"{field}\"]" not in call, (
                f"testrun passes dataset field {field!r} into g.ask: {call}")
        assert "question_date" in call or "as_of" not in call
