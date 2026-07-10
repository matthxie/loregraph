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


def test_prompts_do_not_quote_test_instances():
    """Prompt/schema text must never quote or paraphrase a benchmark question — examples
    must be INVENTED. A rule reverse-engineered from a failing question is fine; naming
    that question's subject in the prompt is the answer key leaking in (it happened:
    'vintage films vs vintage cameras', 'jewelry -> crystal ornament', 'taxi costs $60'
    were all live test instances). Checks every content-word bigram shared between the
    reader prompt / answer schemas and any small-tier question; extend ALLOWED_BIGRAMS
    only for collisions that are genuinely generic phrasing, never for examples."""
    import json as _json
    import os

    from kg.rag import _ANSWER_TOOL, _ANSWER_TOOL_EVENTS, _RAG_SYS

    qpath = os.path.join("dataset", "longmemeval", "small", "questions.jsonl")
    if not os.path.exists(qpath):
        return                                    # tier not built on this machine
    stop = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "for", "with", "from",
            "this", "that", "what", "how", "who", "when", "where", "why", "which", "is",
            "are", "was", "were", "did", "does", "do", "i", "you", "he", "she", "it",
            "my", "your", "their", "our", "at", "by", "as", "be", "been", "has", "have",
            "had", "not", "no", "me", "we", "they", "them", "his", "her", "its", "if",
            "but", "so", "than", "then", "there", "about", "into", "over", "per",
            "many", "much", "long", "ago", "since", "days", "weeks", "months", "years",
            "day", "week", "month", "year", "last", "next", "first", "before", "after",
            "between", "each", "every", "only", "still", "currently", "now", "new",
            "same", "one", "two", "three", "more", "most", "any", "all", "some", "few",
            "need", "get", "go", "take", "make", "say", "state", "stated", "question"}

    def bigrams(text: str) -> set:
        words = [w for w in re.findall(r"[a-z']+", text.lower()) if w not in stop]
        return {f"{a} {b}" for a, b in zip(words, words[1:])}

    def _descriptions(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "description" and isinstance(v, str):
                    yield v
                else:
                    yield from _descriptions(v)
        elif isinstance(node, list):
            for v in node:
                yield from _descriptions(v)

    prompt_text = "\n".join([_RAG_SYS, *_descriptions(_ANSWER_TOOL),
                             *_descriptions(_ANSWER_TOOL_EVENTS)])
    prompt_bg = bigrams(prompt_text)
    # Unigrams are only checked inside the prompt's EXAMPLE spans (quoted or
    # parenthesized text) — that is where a copied question leaks in; rule prose shares
    # ordinary English with questions ("cost", "items") without meaning anything.
    example_spans = re.findall(r'"[^"]{4,}"|\([^)]{4,}\)', prompt_text)
    example_words = {w for span in example_spans
                     for w in re.findall(r"[a-z'-]+", span.lower()) if w not in stop}
    # Generic English that legitimately appears in invented examples; NEVER add a word
    # here to make a copied test-question example pass — invent a new example instead.
    ALLOWED: set = {"would", "cost", "end", "gift", "amount"}
    questions = [_json.loads(line) for line in open(qpath, encoding="utf-8")]
    # a word is DISTINCTIVE if it appears in at most 2 of the tier's questions — a prompt
    # example naming one ("jewelry", "taxi") is almost certainly copied from that question
    from collections import Counter
    freq = Counter()
    qwords = []
    for q in questions:
        ws = {w for w in re.findall(r"[a-z']+", q["query"].lower()) if w not in stop}
        qwords.append(ws)
        for w in ws:
            freq[w] += 1
    offenders = {}
    for q, ws in zip(questions, qwords):
        hit = (bigrams(q["query"]) & prompt_bg) - ALLOWED
        hit |= {w for w in ws & example_words if freq[w] <= 2} - ALLOWED
        if hit:
            offenders[q["query"][:60]] = sorted(hit)
    assert not offenders, (
        "prompt/schema text shares distinctive phrasing with test question(s) — "
        f"replace the example with an invented one: {offenders}")
