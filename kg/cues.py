"""Cue detection — decides which entries earn a paid LLM pass.

The design's input path is local-first ($0) with an LLM call ONLY on entries that
carry signal a local NLP extractor cannot capture cheaply and correctly:

  * termination cues  — a relationship ENDED ("former", "no longer", "quit", "broke up")
  * relative/vague dates — need resolving against the entry's event time ("last March")
  * descriptive identity — references that must bind across entries ("my new manager")

A regex screen is intentionally cheap and high-recall: a false-positive just spends
one extra Haiku call; a false-negative loses temporal/identity signal, so we err
toward escalating. Cue kinds are returned for logging/analysis.
"""
from __future__ import annotations

import re

# A relationship ending / changing state.
_TERMINATION = re.compile(
    r"\b("
    r"former(ly)?|ex-|no longer|used to|no more|"
    r"quit|resigned|left (my|the|his|her|their)|laid off|fired|"
    r"broke up|broken up|split up|divorc\w*|separat\w+|"
    r"moved (out|away|to|from)|relocat\w+|"
    r"stopped|ended|cancell\w+|dropped|gave up|"
    r"until|no longer with"
    r")\b", re.IGNORECASE)

# Relative / vague dates that need anchoring to the event time.
_REL_DATE = re.compile(
    r"\b("
    r"yesterday|today|tomorrow|tonight|last night|this morning|"
    r"last (week|month|year|spring|summer|fall|autumn|winter|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"next (week|month|year)|"
    r"\d+\s+(day|week|month|year)s?\s+ago|"
    r"a (week|month|year) ago|"
    r"back (then|in)|earlier this|recently|these days|nowadays|"
    r"since (last|then)|"
    r"in (19|20)\d\d"
    r")\b", re.IGNORECASE)

# Descriptive identity references that need cross-entry binding (possessive + role).
_IDENTITY = re.compile(
    r"\bmy\s+(new\s+|former\s+|ex[- ]?|current\s+|old\s+)?"
    r"(boss|manager|supervisor|colleague|coworker|co-worker|partner|"
    r"girlfriend|boyfriend|wife|husband|spouse|fiance\w*|landlord|"
    r"roommate|flatmate|doctor|therapist|teacher|professor|mentor|"
    r"client|employer|company|team|friend|neighbou?r)\b", re.IGNORECASE)

# Money amounts and explicit numeral + measurement units — the signal a SUM/total question
# needs, which the local NLP floor doesn't extract as a typed, summable fact. Deliberately
# narrow: a currency symbol or an explicit unit word must be attached to the digits, so bare
# numbers, years ("in 1995"), clock times ("at 5pm"), and ordinals never match — those are
# cheap to mishandle (every one costs an escalation) and are covered, if at all, by
# `_REL_DATE`. "3 of my friends" is a bare count with no unit and intentionally does NOT match.
_CURRENCY_SYMBOL = r"[$€£¥₹]"
_MONEY_WORD = r"dollars?|bucks|cents?|usd|euros?|quid"
_MEASURE_UNIT = (
    r"lbs?|pounds?|kilograms?|kg|grams?|ounces?|oz|"
    r"miles?|kilometers?|km|meters?|feet|ft|inch(?:es)?|"
    r"liters?|litres?|gallons?|quarts?|pints?|dozen(?:s)?"
)
_QUANTITY = re.compile(
    r"(?:"
    rf"{_CURRENCY_SYMBOL}\s?\d[\d,]*(?:\.\d+)?"          # $1,200 / $ 7.5
    rf"|\b\d[\d,]*(?:\.\d+)?\s*(?:{_MONEY_WORD})\b"      # 1200 dollars / 20 bucks
    rf"|\b\d[\d,]*(?:\.\d+)?\s*(?:{_MEASURE_UNIT})\b"    # 10 lbs / 3 miles / 2 dozen
    r")", re.IGNORECASE)

_KINDS = (("termination", _TERMINATION), ("rel_date", _REL_DATE), ("identity", _IDENTITY),
          ("quantity", _QUANTITY))


def cue_kinds(text: str) -> set[str]:
    t = text or ""
    return {name for name, rx in _KINDS if rx.search(t)}


def has_cue(text: str) -> bool:
    t = text or ""
    return any(rx.search(t) for _, rx in _KINDS)
