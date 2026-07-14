"""Regression tests for Finding 8: 'past'/'once' termination folding in kg.extractors.

'past' and 'once' were dropped from the termination-prefix regex, so a label like
'past_employer' no longer folded to employer+ended — it minted a distinct, permanently
open 'past_employer' predicate that never merged with an open 'employer' fact, making an
as-of-now query report a past employer as current. They are re-admitted GUARDED: they fold
only when the remainder is a recognized relation predicate, so the documented false
positives ('past_month', 'past_project', 'once_met') still flow through as asserted.
"""
from kg.extractors import _parse_tool_payload, _normalize_termination, _strip_term_prefix


def _rel(label: str):
    ext = _parse_tool_payload({
        "entities": [{"name": "A", "type": "person"}, {"name": "B", "type": "person"}],
        "tags": [],
        "relations": [{"source": "A", "target": "B", "labels": [label]}],
    })
    return ext.relations[0]


def test_past_employer_folds_to_employer_ended():
    """The finding's canonical case: 'past_employer' -> employer with status=ended, so it
    is mergeable with (and closes) an open 'employer' fact instead of minting a distinct,
    permanently-open predicate."""
    r = _rel("past_employer")
    assert r.status == "ended"
    assert r.labels == ["employer"]


def test_once_prefix_folds_recognized_predicate():
    """'once' also re-admitted when the remainder is a recognized relation predicate."""
    r = _rel("once_colleague")
    assert r.status == "ended"
    assert r.labels == ["colleague"]


def test_documented_false_positives_stay_asserted():
    """The exact concern the removed comment worried about must remain covered: a 'past'/
    'once' marker over a NON-relation remainder (a time period, a thing, a bare verb) is
    NOT a termination and must flow through unchanged as asserted."""
    for label in ("past_month", "past_project", "once_met", "once_lived_in"):
        r = _rel(label)
        assert r.status == "asserted", label
        assert r.labels == [label], label


def test_unambiguous_markers_still_close():
    """The pre-existing unambiguous former-markers are unaffected."""
    assert _rel("former_colleague").status == "ended"
    assert _rel("ex-coworker").status == "ended"
    assert _rel("no_longer_works_with").status == "ended"
    assert _rel("no_longer_works_with").labels == ["works_with"]


def test_strip_term_prefix_unit():
    assert _strip_term_prefix("past_employer") == ("employer", True)
    assert _strip_term_prefix("past_project") == ("past_project", False)
    assert _strip_term_prefix("once_met") == ("once_met", False)
    assert _strip_term_prefix("former_colleague") == ("colleague", True)


def test_normalize_termination_mixed_labels():
    """A relation carrying both a guarded-foldable and a plain label ends up ended, with
    the foldable label reduced to its base predicate and the plain one preserved."""
    labels, ended = _normalize_termination(["past_employer", "works_at"])
    assert ended is True
    assert labels == ["employer", "works_at"]
