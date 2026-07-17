"""Deterministic date-arithmetic pre-judge (kg/testrun.py _date_arith_prejudge).

Motivation: gpt-4o failed the SAME correct "30 days" answer (Jan 2 → Feb 1 2023,
which IS 30 days) in two consecutive runs — the LLM judge does calendar math wrong.
These tests are fully offline: the pre-judge is pure parsing, and the fall-through
path uses a fake OpenAI-shaped judge client.
"""
import json
from types import SimpleNamespace

from kg.metering import UsageMeter
from kg.testrun import _date_arith_prejudge, _judge

_Q_DAYS = {
    "id": "q1",
    "query": "How many days passed between my first church visit and the cathedral trip?",
    "answer": "30 days. 31 days (including the last day) is also acceptable.",
}


def test_correct_day_count_prejudged_without_llm():
    v = _date_arith_prejudge(_Q_DAYS, "The gap was 30 days.")
    assert v is not None and v["correct"] and v["score"] == 1.0
    assert v["method"] == "date_arith"


def test_any_reference_alternative_accepted():
    assert _date_arith_prejudge(_Q_DAYS, "31 days")["correct"]


def test_plus_minus_one_day_tolerance():
    assert _date_arith_prejudge(_Q_DAYS, "29 days")["correct"]      # 30 - 1
    assert _date_arith_prejudge(_Q_DAYS, "32 days")["correct"]      # 31 + 1
    v = _date_arith_prejudge(_Q_DAYS, "About 27 days, I think.")
    assert v is not None and not v["correct"] and v["score"] == 0.0
    assert v["method"] == "date_arith"


def test_coarser_units_exact_only():
    q = {"query": "How many weeks ago did I move?", "answer": "5 weeks"}
    assert _date_arith_prejudge(q, "5 weeks ago")["correct"]
    assert not _date_arith_prejudge(q, "6 weeks ago")["correct"]


def test_non_date_arith_question_falls_through():
    q = {"query": "Where does Becky live?", "answer": "Berlin"}
    assert _date_arith_prejudge(q, "Berlin, since 30 days ago") is None


def test_unparseable_answer_falls_through_to_llm_judge():
    # no count in the model answer → prejudge abstains → the (fake) LLM judge is used
    assert _date_arith_prejudge(_Q_DAYS, "About a month, from early January.") is None

    grade = SimpleNamespace(name="grade", arguments=json.dumps(
        {"correct": True, "score": 1.0, "reason": "ok"}))
    msg = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=[SimpleNamespace(function=grade)]))],
        usage=None)

    class _FakeJudge:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw):
            self.calls.append(kw)
            return msg

    fake = _FakeJudge()
    v = _judge(fake, "gpt-4o", _Q_DAYS, "About a month, from early January.", UsageMeter())
    assert fake.calls and v == {"correct": True, "score": 1.0, "reason": "ok"}
    assert "method" not in v          # only deterministic verdicts carry the marker


def test_no_unit_overlap_falls_through():
    # reference in days, answer only in months → no confident deterministic verdict
    assert _date_arith_prejudge(_Q_DAYS, "Roughly 1 month.") is None
