"""4-lane query router (deterministic, $0).

Classifies a question so retrieval spends effort where it pays:

  RECENCY   — "what did I do yesterday"      → lean on event-time / recent episodes
  STATE     — "who's my manager now", "where  → pull facts (current or as-of-T) and make
               did I live in 2022", "how has    sure the fact-bearing episodes surface;
               X evolved"                       render full closed+open history for evolution
  MULTIHOP  — "who did I talk to about the    → widen the PPR pool (connect-the-dots)
               database project"
  SINGLE    — everything else                → hybrid retrieve + rerank, no special path

The router only nudges retrieval emphasis; the same machine answers every lane, so a
misroute degrades gracefully rather than failing.
"""
from __future__ import annotations

import re

RECENCY = "recency"
STATE = "state"
MULTIHOP = "multihop"
SINGLE = "single"

_RECENCY = re.compile(
    r"\b(yesterday|today|tonight|last night|this (morning|afternoon|evening|week)|"
    r"just now|right now|currently doing|recently|lately)\b", re.IGNORECASE)

_EVOLUTION = re.compile(
    r"\b(evolv\w+|chang\w+ over time|how (has|have|did).*(chang|evolv|progress|develop)|"
    r"over the (past|last) (year|month|few)|trajector\w+|history of|timeline)\b", re.IGNORECASE)

_STATE = re.compile(
    r"\b(now|currently|these days|as of|at present|still|"
    r"who (is|are|was|were) my|where (do|did|does) .* (live|work)|"
    r"what (is|was) my|back in|in (19|20)\d\d|"
    r"used to|no longer|anymore)\b", re.IGNORECASE)

_MULTIHOP = re.compile(
    r"\b(who (did|have) i (talk|speak|meet|discuss|work)|"
    r"connect\w*|related to|because of|why did|"
    r"through whom|introduced (me|by)|"
    r"(everyone|anyone|people|all the .*) (i|who))\b", re.IGNORECASE)

# Date-arithmetic questions ("how many days/months since/between/had passed", "how long
# ago") need the STATE lane's dated-fact emphasis, NOT the aggregation path — checked
# before _AGGREGATE because they also start with "how many". Generic question forms
# honestly available from query text alone.
_DATE_ARITH = re.compile(
    r"\bhow (many|much) (day|week|month|year)s?\b"
    r"|\bhow long (ago|since|had|have|has|was|did|been)\b"
    r"|\b(days?|weeks?|months?|years?) (ago|since|passed|between|had passed|"
    r"have passed|elapsed)\b"
    r"|\bwhen (did|was) (i|my|we)\b", re.IGNORECASE)

# Aggregation questions ("how many X", "how much did I spend", totals/rates) aggregate
# occurrences scattered across sessions — the MULTIHOP (connect-the-dots) lane.
_AGGREGATE = re.compile(
    r"\bhow (many|much|often)\b"
    r"|\b(in total|altogether|combined|overall)\b"
    r"|\btotal (number|amount|cost|hours|time|count)\b"
    r"|\b(per|each|every) (day|week|month|year)\b", re.IGNORECASE)


def route(query: str) -> str:
    """Classify a question into a retrieval lane from its TEXT ALONE. This is the
    production router; it must never see benchmark metadata (question kind/type, gold
    labels) — see tests/test_no_oracle.py, which enforces the signature."""
    q = query or ""
    if _EVOLUTION.search(q):
        return STATE                 # evolution is a STATE-history sub-case
    if _DATE_ARITH.search(q):
        return STATE                 # dated-fact arithmetic, before the "how many" check
    if _AGGREGATE.search(q):
        return MULTIHOP              # count/sum across scattered occurrences
    if _MULTIHOP.search(q):
        return MULTIHOP
    if _STATE.search(q):
        return STATE
    if _RECENCY.search(q):
        return RECENCY
    return SINGLE
